---
title: "NOC 学习路线"
excerpt: "NOC学习项目，含基础与进阶"
collection: portfolio
permalink: /portfolio/noc-study/
project_key: noc-study
---

## 相关文章
{% assign related_posts = site.posts | where: "project", page.project_key | sort: "date" | reverse %}

{% if related_posts.size > 0 %}
{% for post in related_posts %}
- [{{ post.title }}]({{ post.url | relative_url }}) - {{ post.date | date: "%Y-%m-%d" }}
{% endfor %}
{% endif %}