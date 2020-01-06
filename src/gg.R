#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-

library(ggplot2)
library(tools)
library(extrafont)
library(reshape2)
library(optparse)
library(plyr)
font_install('fontcm')
loadfonts()

plot_evolution <- function(direct,ideal_ticks = 10){
  # read in main data
  init_df = read.csv(paste0(direct,"/init.csv"))
  log_df = read.csv(paste0(direct,"/log.csv"))
  model = gsub("(.*_)(.*)(_.*)$","\\2",direct)
  data = gsub("(.*_)(.*)(_(.*))$","\\4",direct)
  # reprocess data with batch visits
  log_df = log_df[c("epoch","d_loss","g_loss")]
  log_df = melt(log_df,id.vars = "epoch",value.name="loss",variable.name="Model")
  log_df = aggregate(.~epoch+Model,log_df,mean,na.action=na.pass)
  max_tick_x = round(max(log_df["epoch"],na.rm = TRUE),-2)
  max_tick_y = round(max(log_df["loss"],na.rm = TRUE),1)
  tick_interval_x <- round_any(max_tick_x/ideal_ticks,50,ceiling)
  tick_interval_y <- round_any(max_tick_y/ideal_ticks,0.5,ceiling)
  # make basic plot
  pdf(paste0(direct,"/vis/evolution.pdf"), width=14, height=7)
  g <- ggplot() +
    geom_line(data=log_df,aes(x=epoch, y=loss, colour=Model),size=0.5) +
    xlab("\nEpoch") + ylab("Batch-Averaged Cross Entropy Loss\n") +
    theme_bw() +
    theme(text = element_text(size=13, family="CM Roman"),
          legend.text=element_text(size=10),
          legend.title=element_text(size=10,face = "bold"),
          legend.key = element_rect(colour = "lightgray", fill = "white"),
          plot.title = element_text(hjust=0.5)) +
    ggtitle(paste0(model," Training Loss Evolution (",toTitleCase(data),")")) +
    scale_color_discrete(breaks=c("g_loss","d_loss"),
                        labels=c("Generator","Discriminator")) +
    scale_x_continuous(breaks=seq(0,max_tick_x,tick_interval_x)) +
    scale_y_continuous(breaks=seq(0,max_tick_y,tick_interval_y))
  # if model is combined, add additional visualizations
  if("until" %in% names(init_df)){
    # recursively find all missing data blocks
    na_epochs <- unique(log_df[which(is.na(log_df[,3])),c("epoch")])
    breaks <- split(na_epochs,cumsum(c(1,diff(na_epochs) != 1)))
    breaks <- do.call(rbind.data.frame,lapply(breaks,function(x) c(min(x),max(x))))
    names(breaks) <- c("xmin","xmax")
    if(nrow(breaks) == 1 & breaks[1,1] == Inf & breaks[1,2] == -Inf){
      plot_breaks <- FALSE
    } else {
      plot_breaks <- TRUE
      breaks$"xmin" <- breaks$"xmin"-1
    }
    until = init_df$until
    until = until[-length(until)]
    g <- g + {if(plot_breaks)geom_rect(data = breaks,aes(xmin=xmin,xmax=xmax,ymin=-Inf,ymax=Inf,fill="Missing data"),alpha=0.4)} +
      {if(length(until) != 0)geom_vline(aes(xintercept=until,linetype="dashed"),size=0.3)} +
      scale_fill_manual(name="",values="lightblue") +
      scale_linetype_manual(name="",values="dashed",labels="Training\nbreak point") +
      guides(color=guide_legend(order=1),fill=guide_legend(order=2))
  }
  # print object
  print(g)
  dev.off()
  # embed latex CM modern
  embed_fonts(paste0(direct,"/vis/evolution.pdf"),
              outfile=paste0(direct,"/vis/evolution.pdf"))
}

parser = OptionParser()
parser <- add_option(parser,c("-d", "--dir"), type="character",
                     help="full path to pickle directory", metavar="directory")
parser <- add_option(parser,c("-t", "--ticks"), type="integer", default=10,
                     help="number of ticks to display", metavar="number of ticks")
opt = parse_args(parser)
plot_evolution(opt$dir,opt$ticks)
