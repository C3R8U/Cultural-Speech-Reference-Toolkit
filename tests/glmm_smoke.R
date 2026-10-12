# Artificial binary fixture solely to test R execution, not paper evidence.
set.seed(17)
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=1) stop('Expected scratch directory')
root <- args[1];dir.create(root,recursive=TRUE,showWarnings=FALSE)
d <- expand.grid(model=c('a','b','c'),seed=1:5,sample_id=1:200,stringsAsFactors=FALSE)
seed_effect <- c(-1.1,-0.6,0,0.6,1.1)
utterance_effect <- rnorm(200,sd=1)
fixed <- c(a=-0.3,b=0.2,c=0.8)
d$correct <- rbinom(nrow(d),1,plogis(fixed[d$model]+seed_effect[d$seed]+utterance_effect[d$sample_id]))
write.csv(d,file.path(root,'artificial_input.csv'),row.names=FALSE)
script <- file.path('statistics','intent_glmm.R')
executable <- if(.Platform$OS.type=='windows') 'Rscript.exe' else 'Rscript'
status <- system2(file.path(R.home('bin'),executable),c(shQuote(script),shQuote(file.path(root,'artificial_input.csv')),shQuote(file.path(root,'output'))))
if(status!=0) stop('GLMM fixture failed')
stopifnot(file.exists(file.path(root,'output','lrt.csv')),
          file.exists(file.path(root,'output','pairwise_holm.csv')))
cat('GLMM_ARTIFICIAL_FIXTURE_PASS\n')
