# Distill 使用说明

`distill/run.sh` 用于批量调用 `nanobot` 处理一个 JSONL 数据集，并将每条样本的会话轨迹保存到工作区的 `sessions/` 目录下。

按以下步骤得到模型的thinking和tool call轨迹

## 配置config.json
基本就是nanobot原始的config，但可以自定义使用哪些tools和skills，比如：
```json
{
  "agents": {
    "defaults": {
      "workspace": "workspaces/qwen35_ptu/test_ptu",
      "model": "qwen3.5-397b-a17b",
      "provider": "gpt_proxy",
      "maxTokens": 32768,
      "temperature": 1.0,
      "maxToolIterations": 80,
      "memoryWindow": 100,
      "reasoningEffort": null
    }
  },
  "channels": {
    "xxx": "xxx",  // 默认的channels
  },
  "providers": {
    "gpt_proxy": {
      "apiKey": "353e7e01f88f7bd033e9b09a74d9d814",
      "apiBase": null,
      "extraHeaders": null
    }
  },
  "gateway": {
    "xxx": "xxx",  // 默认的gateway
  },
  "tools": {
    "web": {
      "proxy": null,
      "search": {
        "apiKey": "",
        "maxResults": 5
      }
    },
    "exec": {
      "timeout": 60,
      "pathAppend": ""
    },
    "restrictToWorkspace": false,
    "mcpServers": {},
    // 这里往下是自定义启用/禁用的tools和skills，会对应修改system prompt
    "enabledTools": {
      "read_file": true,
      "write_file": true,
      "edit_file": true,
      "list_dir": true,
      "exec": true,
      "web_search": false,
      "web_fetch": false,
      "message": false,
      "spawn": false,
      "cron": false,
      "crop": true,
      "draw_bbox": true,
      "draw_line": true,
      "draw_circle": true,
      "color_segments": true,
      "sample_color": true,
      "color_clusters": true,
      "rotate": true,
      "flip": true,
      "enhance_contrast": true,
      "adjust_brightness": true,
      "detect_edges": true,
      "grayscale": true,
      "connected_components": true,
      "find_contours": true,
      "hough_lines": true,
      "hough_circles": true,
      "template_match": true
    },
    "enabledSkills": {
      "memory": false,
      "history": false,
      "clawhub": true,
      "cron": true,
      "github": true,
      "skillCreator": true,
      "summarize": true,
      "tmux": true,
      "weather": true
    }
  }
}
```

## 准备输入文件

输入文件需要是 JSONL 格式，每一行都是一个 JSON object。脚本当前依赖以下字段：

- `id`: 每条样本的唯一标识，用于生成会话名
- `query`: 发送给 `nanobot` 的文本问题
- `images`: 可选，图片路径字符串或字符串列表

说明：

- 如果 `images` 中使用相对路径，会基于 `--media-dir` 进行解析。
- `--query-field` 指定的字段必须存在。

多模态输入示例：

```json
{"id": 0, "query": "How many exajoules did China use in 2019?", "images": ["0.jpg"]}
```

纯文本输入示例：
```json
{"problem": "Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $97_b.$", "answer": 70, "id": "0"}
```

其他文件格式输入示例，注意要设置参数 `--media_field files`：
```json
{"id": "0", "query": "阅读这篇论文，然后简单介绍方法", "files": ["2603.13224v1.pdf"]}
```

## 运行方式

在仓库根目录执行：

```bash
sh distill/run.sh
```

```bash
python distill/run.py \
    dataset/ThinkMorph/Chart_Refocus_6k.jsonl \
    --query-field query \
    --media-field images \
    --media-dir dataset/images/Chart_Refocus_6k \
    --config workspaces/gemini31_pro/config.json \
    --num-workers 1
```

纯文本query示例
```bash
python distill/run.py \
    dataset/AIME25/test.jsonl \
    --query-field problem \
    --config workspaces/gemini31_pro/config.json \
    --num-workers 1
```

其他文件格式示例
```bash
python distill/run.py \
    dataset/test_pdf.jsonl \
    --query-field query \
    --media-field files \
    --media-dir dataset/ \
    --config workspaces/qwen35_ptu/config.json \
    --num-workers 1
```

## 输出位置

运行完成后，会话轨迹会保存到config文件中"workspace"下的 `sessions/` 目录，如

每个query的结果会存在单独的一个session_{id}目录下，包括:
- jsonl文件: 保存system, user, assistant, tool的交互轨迹
- 模型调用tool生成的文件

`visualize.sh`是一个可视化脚本，只需要指定sessions目录

## 参数说明

- `--dataset-name`: 会话名前缀；不传时默认使用输入文件所在目录名
- `--query-field`: 文本问题字段名
- `--media-field`: 图片或其他文件字段名
- `--media-dir`: 相对文件路径的根目录
- `--config`: `nanobot` 配置文件
- `--num-workers`: 并行 worker 数
