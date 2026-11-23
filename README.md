# ai-bg-remover

> 免费下载与图文教程：<https://www.mrheedu.com/apps/ai-bg-remover>　·　更多免费软件：[www.mrheedu.com](https://www.mrheedu.com)
AI去背景工具是一款本地运行的智能抠图软件，内置 U²-Net 深度学习模型与纯色幕布去背双引擎。加载图片或视频后，左侧预览框即时呈现去背结果，透明区域以棋盘格显示；人物、商品、复杂背景交给 AI 自动抠图，绿幕、蓝幕、纯色底素材则可在画面上直接点选颜色，配合容差与边缘参数快速去除，速度约为 AI 的 20 倍。支持透明背景、指定纯色、背景图片、背景视频四种合成方式，图片输出透明 PNG，视频输出 H.264 MP4。全程离线，文件不上传云端，无需注册，永久免费。 本应用已开源（GPL-3.0）：https://gitee.com/FC1976/ai-bg-remover（GitHub 镜像：https://github.com/fengcong1976/ai-bg-remover）。

## 运行

```bat
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python main.py
```

构建与安装包脚本见仓库内的 `.spec`、`.iss` 与 `build*.bat`。

## 许可

GPL-3.0，详见 LICENSE。第三方组件声明见 THIRD_PARTY_NOTICES.md。

## 仓库说明

本仓库只包含源码，以下运行时资源体积较大，未纳入仓库：

- models/u2net.onnx（U²-Net 抠图模型，约 170MB）：请从发布页或自行准备后放入 models/ 目录。

---

## 更多免费软件

本工具由深圳市创炫互动网络科技有限公司开发，官网提供免费安装包、图文教程，以及更多永久免费的 Windows 软件：**[www.mrheedu.com](https://www.mrheedu.com)**
