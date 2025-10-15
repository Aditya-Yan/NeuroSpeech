# NeuroSpeech

once you clone my branch on the repository:

update base directory in following files:
`AnalysisExamples\rnn_step1_makeTFRecords.ipynb`
`AnalysisExamples\rnn_step2_trainBaselineRNN.ipynb`
`AnalysisExamples\rnn_step3_baselineRNNInference.ipynb`

`NeuralDecoder\neuralDecoder\configs\dataset\speech_release_baseline.yaml`

first three, update to be the location of your `speechbci-main` folder
last one, update only the first section BEFORE `\derived` to be the `speechbci-main` folder location.
in step2, also update `outputDir` variable as well

make sure to download the `competitionData.tar.gz` from dryad and place in the `speechbci-main` directory, unzipped twice to not be a `.tar` or `tar.gz` anymore, so it should just be a folder named `competitionData`

cd into the `speechBCI-main\NeuralDecoder` and run `pip install -e .`. this will install dependencies
