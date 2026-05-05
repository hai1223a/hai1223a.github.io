---
title: "示例项目：Python 学习路线"
excerpt: "一个用来演示项目页自动聚合文章的示例项目。"
collection: portfolio
permalink: /portfolio/python-study-demo/
project_key: python-study-demo
---

这是一个示例项目页，用来演示怎么把“一个项目”下面的多篇学习文章自动聚合到一起。

你以后真正要做的事只有两步：

1. 在项目页里写一个唯一的 `project_key`
2. 在相关文章里写同样的 `project`

## 项目目标

- 完成 Python 基础环境配置
- 学会基本语法和数据结构
- 输出连续的学习笔记

## 相关文章

{% assign related_posts = site.posts | where: "project", page.project_key | sort: "date" | reverse %}

{% if related_posts.size > 0 %}
{% for post in related_posts %}
- [{{ post.title }}]({{ post.url | relative_url }}) - {{ post.date | date: "%Y-%m-%d" }}
{% endfor %}
{% else %}
暂时还没有相关文章。
{% endif %}

## 怎么复用这个结构

如果你以后想做自己的项目，比如“深度学习入门”，可以新建一个 `_portfolio/` 里的文件，把 `project_key` 改成：

```yml
project_key: deep-learning-intro
```

然后在对应的文章里都写：

```yml
project: deep-learning-intro
```

这样这个项目页就会自动收集这些文章。
