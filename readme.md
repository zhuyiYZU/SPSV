Firstly install OpenPrompt https://github.com/thunlp/OpenPrompt
Then copy prompts/knowledgeable_verbalizer.py to Openprompt/openprompt/prompts/knowledgeable_verbalizer.py

example shell scripts:

python fewshot.py --result_file ./output_fewshot.txt --dataset agnews --template_id 0 --seed 144 --shot 1 --verbalizer kpt --calibration

python zeroshot.py --result_file ../output_zeroshot.txt --dataset agnews --template_id 0 --seed 144 --verbalizer kpt --calibration

python fewshot_softpilot.py --result_file ./output_fewshot.txt --dataset agnews --template_id 0 --seed 144 --shot 1 --verbalizer kpt --calibration

Note that the file paths should be changed according to the running environment. 

The datasets are downloadable via OpenPrompt.