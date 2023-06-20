# Gloss Informed Bi-encoders for WSD 

This is the codebase for the paper [Moving Down the Long Tail of Word Sense Disambiguation with Gloss Informed Bi-encoders](https://blvns.github.io/papers/acl2020.pdf). 

![Architecture of the gloss informed bi-encoder model for WSD](https://github.com/facebookresearch/wsd-biencoders/blob/main/docs/wsd_biencoder_architecture.jpg)
Our bi-encoder model consists of two independent, transformer encoders: (1) a context encoder, which represents the target word (and its surrounding context) and (2) a gloss encoder, that embeds the definition text for each word sense. Each encoder is initalized with a pertrained model and optimized independently.

## Dependencies 
To run this code, you'll need the following libraries:
* [Python 3](https://www.python.org/)
* [Pytorch 1.2.0](https://pytorch.org/)
* [Pytorch Transformers 1.1.0](https://github.com/huggingface/transformers)
* [Numpy 1.17.2](https://numpy.org/)
* [NLTK 3.4.5](https://www.nltk.org/)
* [tqdm](https://tqdm.github.io/)

We used the [WSD Evaluation Framework](http://lcl.uniroma1.it/wsdeval/) for training and evaluating our model.

## How to Run 
To train a biencoder model, run `python biencoder.py --data-path $path_to_wsd_data --ckpt $path_to_checkpoint`. The required arguments are: `--data-path`, which is the filepath to the top-level directory of the WSD Evaluation Framework; and `--ckpt`, which is the filepath of the directory to which to save the trained model checkpoints and prediction files. The `Scorer.java` in the WSD Framework data files needs to be compiled, with the `Scorer.class` file in the original directory of the Scorer file.

It is recommended you train this model using the `--multigpu` flag to enable model parallel (note that this requires two available GPUs). More hyperparameter options are available as arguments; run `python biencoder.py -h` for all possible arguments.

To evaluate an existing biencoder, run `python biencoder.py --data-path $path_to_wsd_data --ckpt $path_to_model_checkpoint --eval --split $wsd_eval_set`. Without `--split`, this defaults to evaluating on the development set, semeval2007. The model weights and predictions for the biencoder reported in the paper can be found [here](https://drive.google.com/file/d/1NZX_eMHQfRHhJnoJwEx2GnbnYIQepIQj).

Similar commands can be used to run the frozen probe for WSD (`frozen_pretrained_encoder.py`) and the finetuning a pretrained, single encoder classifier for WSD (`finetune_pretrained_encoder.py`).

## How to Reproduce the results of the paper
In order to reproduce the results, the mention library dependences are needed. One problem is that the "Pytorch Transformers 1.1.0" link goes to the wrong package. The correct package is in [this](https://pypi.org/project/pytorch-transformers/) link and can be installed using pip. For convenience I have included in the repo the 'environment.yml' file that was created from my miniconda environment that I used to reproduce the results of the paper.

Before initiating the training of the model, some extrasteps that we have to do are:
* Download the WSD Evaluation framework from the link provided above.
* Create a folder inside which the model training checkpoints and the evaluation results will be stored
* Compile the Scorer.java that can be found in the "WSD_Evaluation_Framework\Evaluation_Datasets" folder. (javac Scorer.java)
* (OPTIONAL) If you have very limited space in your disk you can remove the "SemCor+OMSTI" part of the dataset that can be found inside the "WSD_Evaluation_Framework\Training_Corpora" folder. For the training process, only the SemCor part is used.

To train the model we use the command as described above. Run `python biencoder.py --data-path $path_to_WSD_Evaluation_Framework_folder --ckpt $path_inside_training_checkpoint_folder_you_created --gloss-bsz 64`. I added the gloss batch size part because the training process was using too much GPU memory for my system. You should adjust it accordingly.

After the training process is finished, there should be a checkpoint file created inside the training_checkpoint folder, named 'best_model.ckpt'.

To evaluate the model on the evaluation datasets we use the command as described above. Run `python biencoder.py --data-path $path_to_WSD_Evaluation_Framework_folder --ckpt $path_inside_training_checkpoint_folder_you_created --eval --split $wsd_eval_set`. The parameter --split defines the part of the evaluation dataset that will be used. The possible options are: 'semeval2007', 'senseval2', 'senseval3', 'semeval2013', 'semeval2015' and 'ALL'. Use them without the ''.
After each run the F1 score of the model is printed on the console and a corresponding predictions.txt file is created inside the training checkpoints folder.

## How to extract word and sense embeddings
In order to extract the word and sense embeddings that the model creates for one of the evaluation datasets, we run `python biencoder2.py --data-path $path_to_WSD_Evaluation_Framework_folder --ckpt $path_inside_training_checkpoint_folder_you_created --extract_embeddings --embeddings_dataset_source eval --split $wsd_eval_set --embeddings_output_format txt --embeddings_output_folder $path_to_output_folder`.
The parameter '--embeddings_dataset_source' specifies that we want to extract the embeddings from an evaluation dataset. The parameter '--embeddings_output_format' defines the type of the output file. Can be either txt or pkl (for pickle file). 

In order to extract the word and sense embeddings that the model creates for the training dataset, we run `python biencoder2.py --data-path $path_to_WSD_Evaluation_Framework_folder --ckpt $path_inside_training_checkpoint_folder_you_created --extract_embeddings --embeddings_dataset_source train --embeddings_output_format txt --embeddings_output_folder $path_to_output_folder --multigpu`.
The '--multigpu' flag is not necessary, I used it for speed and memory issues.

## Citation
If you use this work, please cite the corresponding [paper](https://blvns.github.io/papers/acl2020.pdf):
```
@inproceedings{
  blevins2020wsd,
  title={Moving Down the Long Tail of Word Sense Disambiguation with Gloss Informed Bi-encoders},
  author={Terra Blevins and Luke Zettlemoyer},
  booktitle={Proceedings of the 58th Association for Computational Linguistics},
  year={2020},
  url={https://blvns.github.io/papers/acl2020.pdf}
}
```

## Contact
Please address any questions or comments about this codebase to blvns@cs.washington.edu. If you want to suggest changes or improvements, please check out the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
This codebase is Attribution-NonCommercial 4.0 International licensed, as found in the [LICENSE](https://github.com/facebookresearch/wsd-biencoders/blob/master/LICENSE) file.
