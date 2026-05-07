---
title: "论文学习"
excerpt:
collection: portfolio
permalink: /portfolio/paper-study/
project_key: paper-study
---

{% assign related_posts = site.posts | where: "project", page.project_key | sort: "date" | reverse %}

{% if related_posts.size > 0 %}
{% for post in related_posts %}
- [{{ post.title }}]({{ post.url | relative_url }}) - {{ post.date | date: "%Y-%m-%d" }}
{% endfor %}
{% endif %}
