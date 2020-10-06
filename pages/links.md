---
layout: page
title: Links
description: 学习常用网站
keywords: 链接
comments: false
menu: 链接
permalink: /links/
---

> 没有灵感？看论文吧

<ul>
{% for link in site.data.links %}
  {% if link.src == 'paper' %}
  <li><a href="{{ link.url }}" target="_blank">{{ link.name}}</a></li>
  {% endif %}
{% endfor %}
</ul>

> 看看别人在搞啥

<ul>
{% for link in site.data.links %}
  {% if link.src == 'other' %}
  <li><a href="{{ link.url }}" target="_blank">{{ link.name}}</a></li>
  {% endif %}
{% endfor %}
</ul>

> 学习资源

<ul>
{% for link in site.data.links %}
  {% if link.src == 'learn' %}
  <li><a href="{{ link.url }}" target="_blank">{{ link.name}}</a></li>
  {% endif %}
{% endfor %}
</ul>
