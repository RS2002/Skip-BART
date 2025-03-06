# ML-BART
**Article:** "Is Stage Lighting the Inspiration for Art or Mechanized Formula: A BART Inspired Approach" (under way)



## 1. Model Structure

![](./img/model.png)



![](./img/workflow.png)

## 2. Dataset: RPMC-L

[RPMC_L2](https://zenodo.org/records/14854217?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjM5MDcwY2E5LTY0MzUtNGZhZC04NzA4LTczMjNhNTZiOGZmYSIsImRhdGEiOnt9LCJyYW5kb20iOiI1YWRkZmNiMmYyOGNiYzI4ZWUxY2QwNTAyY2YxNTY4ZiJ9.0Jr6GYfyyn02F96eVpkjOtcE-MM1wt-_ctOshdNGMUyUKI15-9Rfp9VF30_hYOTqv_9lLj-7Wj0qGyR3p9cA5w)



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

