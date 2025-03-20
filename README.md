# Skip-BART
**Article:** Zijian Zhao¹, Dian Jin¹, Zijing Zhou¹, Xiaoyu Zhang*, "Is Stage Lighting the Inspiration for Art or Mechanized Formula" (under way)

¹: equal contributions ([Tokamak Disruption](https://tokamak-disruption.netlify.app/)), *: corresponding author

**Some parts of the code is based on:** 
[RS2002/Adversarial-MidiBERT: Official Repository for The Paper, Let Network Decide What to Learn: Symbolic Music Understanding Model Based on Large-scale Adversarial Pre-training](https://github.com/RS2002/Adversarial-MidiBERT)
[RS2002/PianoBart: Official Repository for The Paper, PianoBART: Symbolic Piano Music Understanding and Generating with Large-Scale Pre-Training](https://github.com/RS2002/PianoBart)
[RS2002/CSI-BERT: Official Repository for The Paper, Finding the Missing Data: A BERT-inspired Approach Against Package Loss in Wireless Sensing](https://github.com/RS2002/CSI-BERT)
[RS2002/CSI-BERT2: Official Repository for The Paper, CSI-BERT2: A BERT-Inspired Framework for Efficient CSI Prediction and Recognition in Wireless Communication and Sensing](https://github.com/RS2002/CSI-BERT2)



## 1. Model Structure

<img src="./img/model.png" style="zoom:50%;" />



![](./img/workflow.png)



## [2. Dataset: RPMC_L2](https://zenodo.org/records/14854217?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjM5MDcwY2E5LTY0MzUtNGZhZC04NzA4LTczMjNhNTZiOGZmYSIsImRhdGEiOnt9LCJyYW5kb20iOiI1YWRkZmNiMmYyOGNiYzI4ZWUxY2QwNTAyY2YxNTY4ZiJ9.0Jr6GYfyyn02F96eVpkjOtcE-MM1wt-_ctOshdNGMUyUKI15-9Rfp9VF30_hYOTqv_9lLj-7Wj0qGyR3p9cA5w)



## 3. How to Run

### 3. 1 Pre-train

```shell
python pretrain.py
```



### 3.2 Fine-tune

```shell
python finetune.py --model_path <pre-trained bart path>
```



## 4. Generate Your Own Stage Light

```shell
python generate.py --bart_path <fine-tuned backbone path> --head_path <fine-tuned model head path> --music_file <music file path>
```



## 5. Citation

```

```

