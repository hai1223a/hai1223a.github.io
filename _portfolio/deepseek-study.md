---
title: "DeepSeek 系列文章集"
excerpt: "整理 DeepSeek 系列模型的架构演进与论文阅读笔记，关注注意力机制、KV cache、MoE 及推理实现。"
collection: portfolio
permalink: /portfolio/deepseek-study/
project_key: deepseek-study
---

记录 DeepSeek 系列模型的架构与论文学习笔记，重点关注注意力机制、KV cache、专家组织，以及它们对计算、存储和通信的影响。

## 架构系列

{% assign architecture_posts = site.posts | where: "project", page.project_key | sort: "date" %}
{% for post in architecture_posts %}
- [{{ post.title }}]({{ post.url | relative_url }})
  <br>{{ post.excerpt }}
{% endfor %}

## 相关论文笔记

{% assign series_posts = site.posts | where: "series", page.project_key | sort: "date" %}
{% for post in series_posts %}
{% unless post.project == page.project_key %}
- [{{ post.title }}]({{ post.url | relative_url }})
  <br>{{ post.excerpt }}
{% endunless %}
{% endfor %}
