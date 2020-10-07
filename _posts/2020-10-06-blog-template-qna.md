---
layout: post
title: 本博客常见问题 Q & A
categories: GitHub
description: 使用这个博客时可能会遇到的问题的汇总。
keywords: Jekyll, GitHub Pages
topmost: true
original: true
---

## 是否支持画流程图、时序图、mermaid 和 MathJax

支持。因为相关的引入文件比较大可能影响加载速度，没有默认对所有文件开启，需要在要想开启的文件的 Front Matter 里加上声明：

```yaml
---
flow: true
sequence: true
mermaid: true
mathjax: true
---
```

以上四个开关分别对应 flowchart.js（流程图）、sequence-diagram.js（时序图）、mermaid 和 MathJax 的支持，按需开启即可，然后就可以在正文里正常画图了，展示效果可以参见 <https://mazhuang.org/wiki/markdown/>，对应写法参考源文件 <https://github.com/mzlogin/mzlogin.github.io/blob/master/_wiki/markdown.md>。

## 如何修改代码高亮风格

可以通过 _config.yml 文件里的配置项 `highlight_theme` 来指定代码高亮风格，支持的风格名称列表参考另一个项目：

- <https://github.com/mzlogin/rouge-themes>

在项目主页可以看到每种风格的预览效果。
