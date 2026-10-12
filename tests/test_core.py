import unittest,itertools,math,tempfile,json,sys
from pathlib import Path
import numpy as np
import torch
from culture_speech.model import Config,SpeechSemanticModel,joint_loss,save_checkpoint,load_checkpoint
from culture_speech.decoding import prefix_beam,rerank
from culture_speech.quantize import quantize,load_int8
from culture_speech.statistics import holm,mean_ci,paired_bootstrap,analyze_errors,analyze_ratings
from culture_speech.metrics import edit_distance,error_counts

class CoreTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(17);torch.set_num_threads(1)
    def small_model(self,bridge='dcsb'):
        return SpeechSemanticModel(Config(8,hidden=8,embedding=6,acoustic_layers=1,semantic_layers=1,dropout=0,bridge=bridge))
    def test_semantic_gradients_reach_acoustic_encoder(self):
        model=self.small_model();output=model(torch.randn(2,12,39),torch.tensor([12,9]))
        torch.nn.functional.cross_entropy(output['intent_logits'],torch.tensor([0,2])).backward()
        self.assertGreater(float(model.acoustic.weight_ih_l0.grad.abs().sum()),0)
    def test_hard_cascade_blocks_semantic_acoustic_gradient(self):
        model=self.small_model('hard');output=model(torch.randn(1,12,39),torch.tensor([12]))
        output['intent_logits'].sum().backward();self.assertIsNone(model.acoustic.weight_ih_l0.grad)
    def test_padded_batch_invariance(self):
        model=self.small_model().eval();x=torch.randn(1,9,39)
        alone=model(x,torch.tensor([9]))['representation']
        padded=torch.cat([x,torch.randn(1,5,39)],dim=1)
        batch=torch.cat([padded,torch.randn(1,14,39)],dim=0)
        together=model(batch,torch.tensor([9,14]))['representation'][0]
        torch.testing.assert_close(alone[0],together,rtol=1e-4,atol=1e-5)
    def test_joint_loss_and_generation(self):
        model=self.small_model();features=torch.randn(2,12,39);lengths=torch.tensor([12,9])
        di=torch.tensor([[1,4,5],[0,0,0]])
        batch={'ctc_targets':torch.tensor([4,5,6,4]),'ctc_lengths':torch.tensor([2,2]),'lengths':lengths,
               'intent_labels':torch.tensor([2,0]),'decoder_target':torch.tensor([[4,5,2],[0,0,0]])}
        output=model(features,lengths,di);loss,_=joint_loss(output,batch);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(model.decoder.weight_ih_l0.grad.abs().sum()),0)
        model.eval();generated=model.generate(output['representation'][:1],max_tokens=5,beam_size=3)
        self.assertLessEqual(len(generated),5)
        self.assertTrue(all(k>=3 for k in generated))
    def test_ctc_matches_exhaustive_paths(self):
        probabilities=np.array([[.5,0,0,.3,.2],[.2,0,0,.6,.2],[.4,0,0,.2,.4]])
        merged={}
        for path in itertools.product([0,3,4],repeat=3):
            collapsed=[];previous=None
            for k in path:
                if k!=previous and k!=0:collapsed.append(k)
                previous=k
            key=tuple(collapsed);merged[key]=merged.get(key,0)+np.prod([probabilities[t,k] for t,k in enumerate(path)])
        log_probs=np.full_like(probabilities,-math.inf)
        np.log(probabilities,out=log_probs,where=probabilities>0)
        candidates=prefix_beam(log_probs,beam_size=100)
        self.assertEqual(len(candidates),len(merged))
        for sequence,score in candidates:self.assertAlmostEqual(math.exp(score),merged[tuple(sequence)],places=12)
    def test_int8_forward_checkpoint_roundtrip(self):
        model=self.small_model().eval();q=quantize(model);x=torch.randn(1,12,39);lengths=torch.tensor([12])
        output=q(x,lengths,torch.tensor([[1,4]]))
        self.assertTrue(torch.isfinite(output['intent_logits']).all())
        self.assertTrue(any('quantized' in type(m).__module__ for m in q.modules()))
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            fp=Path(directory)/'fp.pt';save_checkpoint(fp,model,['<blank>','<bos>','<eos>','<unk>','甲','乙','丙','丁'])
            restored,_=load_checkpoint(fp)
            torch.testing.assert_close(model(x,lengths)['intent_logits'],restored.eval()(x,lengths)['intent_logits'])
            path=Path(directory)/'int8.pt'
            torch.save({'config':vars(model.config),'vocab':[],'extra':{},'state_dict':q.state_dict(),
                'torch_version':str(torch.__version__),'engine':torch.backends.quantized.engine},path)
            q2,_=load_int8(path);torch.testing.assert_close(output['intent_logits'],q2(x,lengths)['intent_logits'])
    def test_holm_and_seed_ci(self):
        np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])
        result=mean_ci([1,2,3,4,5]);self.assertEqual(result['mean'],3)
        self.assertAlmostEqual(result['ci95'][1],4.963243,places=5)
    def test_bootstrap_paired_identical_systems(self):
        a=np.array([[1,2,0],[2,1,0]]);n=np.full_like(a,10)
        r=paired_bootstrap(a,a,n,iterations=100)
        self.assertEqual(r['difference_a_minus_b'],0);self.assertEqual(r['p_two_sided_null_centered'],1)
        self.assertEqual(r['ci95'],[0,0])
    def test_bootstrap_rejects_unpaired_and_invalid(self):
        with self.assertRaises(ValueError):paired_bootstrap(np.zeros((2,3)),np.zeros((2,4)),np.ones((2,3)))
        with self.assertRaises(ValueError):paired_bootstrap(np.zeros((2,3)),np.zeros((2,3)),np.zeros((2,3)))
    def test_nonzero_bootstrap_and_error_grid(self):
        a=np.ones((5,3));b=np.full((5,3),2);n=np.full((5,3),10)
        r=paired_bootstrap(a,b,n,iterations=100)
        self.assertAlmostEqual(r['difference_a_minus_b'],-.1);self.assertLess(r['p_two_sided_null_centered'],.02)
        rows=[{'model':model,'seed':str(seed),'sample_id':str(u),'errors':err,
               'reference_units':10,'unit':'character','reference':'甲'*10}
              for model,err in [('a',1),('b',2)] for seed in range(5) for u in range(3)]
        report=analyze_errors(rows,'a',iterations=100);self.assertEqual(report['sentence_count'],3)
        with self.assertRaises(ValueError):analyze_errors(rows[:-1],'a',iterations=100)
    def test_rating_ties_and_unpaired(self):
        rows=[{'unit_id':str(i),'model':m,'score':3} for i in range(4) for m in ['a','b','c']]
        result=analyze_ratings(rows);self.assertEqual(result['p'],1);self.assertEqual(result['kendall_w'],0)
        with self.assertRaises(ValueError):analyze_ratings(rows[:-1])
    def test_character_metric(self):
        self.assertEqual(edit_distance('甲乙丙','甲丁丙'),1)
        self.assertEqual(error_counts('甲，乙。','甲乙'),(0,2))
    def test_scaffolds_not_research_ready(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
        from check_data import inspect
        root=Path(__file__).resolve().parents[1]/'data'/'collection_scaffolds'
        for name,n in [('CSSP',4800),('Field-Culture_Test',600)]:
            report=inspect(root/name,'scaffold');self.assertTrue(report['passed']);self.assertFalse(report['research_ready'])
            self.assertEqual(report['actual_audio'],0);self.assertEqual(report['samples'],n)
            self.assertFalse(inspect(root/name,'research')['passed'])
        from culture_speech.data import SpeechDataset
        with self.assertRaises(ValueError):SpeechDataset(root/'CSSP'/'samples.jsonl',root/'CSSP','train',['<blank>','<bos>','<eos>','<unk>'])
if __name__=='__main__':unittest.main()
