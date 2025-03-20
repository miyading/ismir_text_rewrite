To obtain meta audiobox-aesthetics score, one need to follow the instructions proposed in [this repo](https://github.com/facebookresearch/audiobox-aesthetics?tab=readme-ov-file#usage). In this repo we include some helper scripts to run the evaluation more smoothly.

In `audio_eval.py` file, we provided `create_file_path_jsonl` funciton to output the jsonl file with all audios, and in `combine_results()` function evalutation results are combined with the audio's path.

A typical run of evaluation process goes like:
```shell
python audio_eval.py  #execute create_file_path_jsonl in main
audio-aes audios.jsonl --batch-size 100 > output.jsonl
python audio_eval.py #execute combine_results in main 
```