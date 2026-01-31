# Skip-BART
**Article:** Zijian Zhao, Dian Jin, Zijing Zhou, Xiaoyu Zhang*, "[Automatic Stage Lighting Control: Is it a Rule-Driven Process or Generative Task?](https://openreview.net/forum?id=a4Got6azjF&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions))", 2026 International Conference on Learning Representations (ICLR), 2026

**Notice:** We have uploaded the model, full parameters ([RS2002/Skip-BART · Hugging Face](https://huggingface.co/RS2002/Skip-BART)), and dataset ([RS2002/RPMC-L2 · Datasets at Hugging Face](https://huggingface.co/datasets/RS2002/RPMC-L2)) in Hugging Face.

## 1. Model Structure

<img src="./img/model.png" style="zoom:50%;" />



![](./img/workflow.png)



## 2. Dataset: RPMC-L2

The dataset is available at [RPMC_L2](https://zenodo.org/records/14854217?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjM5MDcwY2E5LTY0MzUtNGZhZC04NzA4LTczMjNhNTZiOGZmYSIsImRhdGEiOnt9LCJyYW5kb20iOiI1YWRkZmNiMmYyOGNiYzI4ZWUxY2QwNTAyY2YxNTY4ZiJ9.0Jr6GYfyyn02F96eVpkjOtcE-MM1wt-_ctOshdNGMUyUKI15-9Rfp9VF30_hYOTqv_9lLj-7Wj0qGyR3p9cA5w). More details can be found at the [paper](https://arxiv.org/abs/2506.01482).



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

The trained parameters are available at [trained.zip](https://huggingface.co/RS2002/Skip-BART/blob/main/trained.zip) .



## 5. Citation

```
@inproceedings{zhao2026automatic,
  title={Automatic Stage Lighting Control: Is it a Rule-Driven Process or Generative Task?},
  author={Zhao, Zijian and Jin, Dian and Zhou, Zijing and Zhang, Xiaoyu},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```

