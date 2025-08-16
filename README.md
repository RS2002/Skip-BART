# Skip-BART
**Article:** Zijian Zhao, Dian Jin, Zijing Zhou, Xiaoyu Zhang*, "[Automatic Stage Lighting Control: Is it a Rule-Driven Process or Generative Task?](https://arxiv.org/abs/2506.01482)" (under review)



## 1. Model Structure

<img src="./img/model.png" style="zoom:50%;" />



![](./img/workflow.png)



## 2. Dataset: RPMC-L2

The dataset is available at [RPMC_L2](https://zenodo.org/records/14854217?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjM5MDcwY2E5LTY0MzUtNGZhZC04NzA4LTczMjNhNTZiOGZmYSIsImRhdGEiOnt9LCJyYW5kb20iOiI1YWRkZmNiMmYyOGNiYzI4ZWUxY2QwNTAyY2YxNTY4ZiJ9.0Jr6GYfyyn02F96eVpkjOtcE-MM1wt-_ctOshdNGMUyUKI15-9Rfp9VF30_hYOTqv_9lLj-7Wj0qGyR3p9cA5w) anonymously.



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

The trained parameters are available at [trained.zip](https://zenodo.org/records/15348791?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImNiMzM5Y2RkLWEyODktNDRkYy1iOTE0LTQ3MmRmMjVjNzljMiIsImRhdGEiOnt9LCJyYW5kb20iOiJhMjkzZjNkYjlkNWM4NTUyOWQwZWJiZWY2MmYzMDFlMCJ9.MamQE2FaJ4LqdbgS9cOeLRZEa559Am8_xbAjwXLX0IRSfNtgvnjcy2G2SEJHYVPX0nNY3-6o7lAbGk39AyyrWA) anonymously.



## 5. Citation

```
@article{zhao2025automatic,
  title={Automatic Stage Lighting Control: Is it a Rule-Driven Process or Generative Task?},
  author={Zhao, Zijian and Jin, Dian and Zhou, Zijing and Zhang, Xiaoyu},
  journal={arXiv preprint arXiv:2506.01482},
  year={2025}
}
```

