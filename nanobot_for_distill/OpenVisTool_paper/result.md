## Main Results

不同模型，都用OpenVisTool-42K训练三个Epoch

| -                    | -         | Chart | GUI  | Table | VisualSearch | Web2HTML | AVG  |
|----------------------|-----------|-------|------|-------|--------------|----------|------|
| Qwen3.5 4B           | w/o tool  | 31.6  | 22.0 | 25.4  | 34.0         | 40.8     | 30.8 |
|  + openvistool 42k   | with tool | 43.0  | 27.4 | 37.3  | 56.6         | 43.4     | 41.5 |
| Qwen3.5 9B           | w/o tool  | 34.6  | 27.1 | 25.8  | 35.6         | 40.1     | 32.6 |
|  + openvistool 42k   | with tool | 49.3  | 31.6 | 43.1  | 59.4         | 45.5     | 45.8 |
| Qwen3.5-27B          | w/o tool  | 45.7  | 28.9 | 29.8  | 43.2         | 46.6     | 38.8 |
|  + openvistool 42k   | with tool | 60.7  | 31.6 | 48.2  | 57.3         | 48.7     | 49.3 |
| Qwen3-VL-8B-Instruct | w/o tool  | 22.1  | 13.0 | 26.7  | 31.8         | 26.5     | 24.0 |
|  + openvistool 42k   | with tool | 26.1  | 19.7 | 35.3  | 50.7         | 32.9     | 32.9 |

## Ablation on Rejection Sampling Strategy

三种拒绝采样策略的消融，以及相同query但用纯文本轨迹训练的对比实验(no tool sft)，所有实验组都训练相同的steps(1971个steps)

| -                  | -         | Chart | GUI  | Table | VisualSearch | Web2HTML | AVG  |
|--------------------|-----------|-------|------|-------|--------------|----------|------|
| Qwen3.5 9B         | w/o tool  | 34.6  | 27.1 | 25.8  | 35.6         | 40.1     | 32.6 |
|  + openvistool 42k | with tool | 49.3  | 31.6 | 43.1  | 59.4         | 45.5     | 45.8 |
|  acc_only          | with tool | 48.6  | 30.8 | 38.9  | 57.3         | 45.0     | 44.1 |
|  toolgain_only     | with tool | 46.8  | 27.8 | 37.5  | 54.7         | 44.9     | 42.4 |
|  no tool sft       | w/o tool  | 42.1  | 40.0 | 30.0  | 30.0         | 43.8     | 37.1 |

## Cross Domain Analysis

观察是否有跨领域泛化，在训练集子集上训3个Epoch

| -               | -         | Chart | GUI  | Table | VisualSearch | Web2HTML | AVG  |
|-----------------|-----------|-------|------|-------|--------------|----------|------|
| Qwen3.5 9B      | w/o tool  | 34.6  | 27.1 | 25.8  | 35.6         | 40.1     | 32.6 |
|  + chart        | with tool | 46.6  | 22.2 | 37.1  | 44.3         | 36.5     | 37.4 |
|  + gui          | with tool | 41.4  | 36.3 | 38.7  | 52.8         | 39.1     | 41.7 |
|  + table        | with tool | 45.7  | 20.5 | 38.9  | 45.8         | 39.2     | 38.0 |
|  + visualsearch | with tool | 45.7  | 23.3 | 40.9  | 57.3         | 21.6     | 37.8 |
|  + web2html     | with tool | 41.6  | 18.2 | 38.5  | 42.2         | 44.6     | 37.0 |
