SITS
====
This is an implementation of the Gibbs samplers for SITS models (both parametric and nonparametric versions) described in Nguyen et al. (ACL 2012). For more information about the model, please refer to the paper.
```
@inproceedings{Nguyen:Boyd-Graber:Resnik-2012,
	Author = {Viet-An Nguyen and Jordan Boyd-Graber and Philip Resnik},
	Booktitle = {Association for Computational Linguistics},
	Year = {2012},
	Location = {Jeju, South Korea},
	Title = {{SITS}: A Hierarchical Nonparametric Model using Speaker Identity for Topic Segmentation in Multiparty Conversations},
}
```

An extended version (with more details and experiments) was published in this article in Machine Learning journal:
```
@article{Nguyen:Boyd-Graber:Resnik:Cai:Midberry:Wang-2014,
	Publisher = {Springer},
	Title = {Modeling Topic Control to Detect Influence in Conversations using Nonparametric Topic Models},
	Booktitle = {Machine Learning},
	Author = {Viet-An Nguyen and Jordan Boyd-Graber and Philip Resnik and Deborah Cai and Jennifer Midberry and Yuanxin Wang},
	Year = {2014},
	Volume = {95},
  	Number = {3},
  	Pages = {381--421},
}
```

# Compile
- To compile: `ant compile`
- To make a clean build: `ant clean-build`
- To makr the jar file: `ant jar`

Please refer to the file `build.xml` for additional options.

# Input Data
SITS takes as inputs a set of conversations, each has multiple turns, each of which is a maximal uninterrupted utterance by one speaker. Currently, SITS accepts the following files:

1. `<dataset>.words`: contains the main texts in the following format:
```
  <num-conversations>\n
  <total-num-turns>\n
  <num-words-conv-1-turn-1>\t<word-1> <word-2> ...\n
  <num-words-conv-1-turn-2>\t<word-1> <word-2> ...\n
  ...\n
  <num-words-conv-1-turn-T1>\t<word-1> <word-2> ...\n
  \n
  <num-words-conv-2-turn-1>\t<word-1> <word-2> ...\n
  <num-words-conv-2-turn-2>\t<word-1> <word-2> ...\n
  ...\n
  <num-words-conv-2-turn-T2>\t<word-1> <word-2> ...\n
```
Here a blank line is used to separate two conversations. Each word is an index in the word vocabulary stored in file `<dataset>.voc`.

2. `<dataset>.show`: contains the conversation name for each turn. The number of lines in this file is equal to the number of turns in `<dataset>.words`

3. `<dataset>.authors`: contains the speaker of each turn. Each speaker is an index in the speaker vocabulary, stored in file `<dataset>.whois`

4. `<dataset>.voc`: contains the word vocabulary

5. `<dataset>.whois`: contains the speaker vocabulary

6. `<dataset>.text`: contains the raw texts

An example of a formatted data is also included in folder `data`.

# Run models
## Parametric SITS
```
java -cp 'dist/sits.jar:lib/*' segmentation.TopicSegmentation --dataset <dataset> --input <format_folder> --output <output_folder> --model param -v
```
Here are the arguments:
- `<dataset>`: name of the dataset, which is also the file name in the formatted folder (see above).
- `<format_folder>`: path to the folder containing the formatted data.
- `<output_folder>`: path to the folder to store the output
- `burnIn`: number of iterations during the burn-in period (default: 2500)
- `maxIter`: maximum number of iterations (default: 5000)
- `sampleLag`: lag between samples (default: 100)
- `K`: number of topics (default: 25)
- `alpha`: Dirichlet parameter for documents' topic distribution (default: 0.1)
- `beta`: Dirichlet parameter for topics' word distribution (default: 0.1)
- `gamma`: Beta parameter for speakers' topic shift distribution (default: 0.25)

Example
```
java -cp 'dist/sits.jar:lib/*' segmentation.TopicSegmentation --dataset debate2008 --input data/debate2008/ldaformat/ --output data/segmentation/debate2008/ --burnIn 100 --maxIter 5000 --sampleLag 50 --gamma 2.5 --model param -v --alpha 0.1 --beta 0.1
```

## Nonparametric SITS
```
java -cp 'dist/sits.jar:lib/*' segmentation.TopicSegmentation --dataset <dataset> --input <format_folder> --output <output_folder> --model non-param -v
```
Here are the arguments:
- `<dataset>`: name of the dataset, which is also the file name in the formatted folder (see above).
- `<format_folder>`: path to the folder containing the formatted data.
- `<output_folder>`: path to the folder to store the output
- `burnIn`: number of iterations during the burn-in period (default: 2500)
- `maxIter`: maximum number of iterations (default: 5000)
- `sampleLag`: lag between samples (default: 100)
- `K`: initial number of topics (default: 25)
- `alpha`, `alpha_0`, `alpha_C`: Dirichlet process parameter for documents' topic distribution (default: 0.1)
- `beta`: Dirichlet parameter for topics' word distribution (default: 0.1)
- `gamma`: Beta parameter for speakers' topic shift distribution (default: 0.25)
