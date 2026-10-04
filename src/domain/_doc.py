# -*- coding: utf-8 -*-
"""把**给同事看的开发注释**压成**给模型看的一行说明**。

文件职责：
    本模块只做一件事 —— 在生成 JSON Schema 时，把类 docstring 的**第一行**
    取出来当 ``description``，丢掉后面的全部内容。

上下游依赖：
    - 上游：仅 ``pydantic`` 的类型标注与标准库。
    - 下游：:mod:`src.domain.enums` 的 :class:`_AligoStrEnum`、
      :mod:`src.domain.schemas` 的 :class:`_Schema`。

═══ 为什么需要它：这不是风格问题，是提示词被污染的问题 ═══

pydantic 生成 JSON Schema 时，会把类的 ``__doc__`` **整段**塞进
``description``。而本项目的文档要求是「详尽解释为什么」，于是一个业务类的
docstring 动辄一两千字，里面含：

* ⚠️ 标记 —— 本项目「这一行是坑」的排版约定；
* ``file:line`` 引用，例如 ``src/orchestration/prompt.py``；
* 反引号包起来的代码片段与配置键；
* 面向**开发者**的告诫，例如「``OTHER`` 必须由代码兜底」。

这些字符串会随 ``structured_schema`` 一起，**每一轮对话**都发给大模型。
后果有两层，第二层更要命：

1. **浪费**。结构化输出的 schema 每轮都要重发，长文档直接抬高 token 成本。
2. **误导**。schema 里的 ``description`` 在模型眼里是**指令**，不是注释。
   一句「``OTHER`` 必须存在且必须由代码兜底」是写给开发者的实现要求，
   模型却会读成「多输出 ``OTHER``」—— 于是这条注释**主动降低了**识别准确率。

结论：给同事看的注释与给模型看的提示词**必须分开**，而分开的动作要在
schema 生成这一层做，因为那里是两者唯一的交汇点。

⚠️ 两条泄漏路径，缺一不可（见 ``tests/test_domain_schemas.py`` 的
   ``test_enum_docstrings_do_not_leak_into_json_schema``）：

   * **枚举**的 docstring → 出现在 ``$defs`` 里该枚举的 ``description``；
   * **模型**的 docstring → 出现在它自己 schema 顶层的 ``description``。

   只堵前者（枚举）是个很容易犯的错：枚举是「被复用的类型」，改一处能
   覆盖很多字段，看起来很划算，于是会让人以为已经处理完了。但模型的
   docstring 走的是另一条路径，实测中它才是内容最长的那一份。
"""

from __future__ import annotations

from pydantic.json_schema import JsonSchemaValue


def first_line_summary(doc: str | None) -> str:
    """取 docstring 的第一行，去掉 Markdown 强调标记。

    约定：本项目每个类 docstring 的第一行都是一句**干净的概括**
    （不含 ⚠️、不含代码引用），因此它天然适合当模型可见的说明。

    Args:
        doc (`str | None`): 类的 ``__doc__``；可能为 ``None``。

    Returns:
        `str`: 可直接写进 JSON Schema 的一行说明；无有效内容时为空串。

    ⚠️ 去掉 ``**`` 与 ``` `` ``` 两个标记，是为了让这句话在 JSON 里保持
    朴素 —— 模型对 Markdown 强调的处理并不可靠，而这两个标记在本项目里
    纯粹是排版手段，没有语义。
    """
    if not doc:
        return ""
    for line in doc.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped.replace("**", "").replace("``", "")
    return ""


def apply_description(json_schema: JsonSchemaValue, doc: str | None) -> JsonSchemaValue:
    """把一行说明写进 schema；没有内容时**删除** description 字段。

    Args:
        json_schema (`JsonSchemaValue`): pydantic 生成的 schema（就地修改）。
        doc (`str | None`): 类的 ``__doc__``。

    Returns:
        `JsonSchemaValue`: 同一个 schema 对象。

    ⚠️ 无 docstring 时**删除**而不是留空串：``""`` 在模型看来仍是一个存在
    的字段说明，会占位；缺失则干净利落。框架自带的类（如
    ``ToolResultState``）都不带 docstring，本项目的类型与它们混用在同一个
    schema 里，两种写法必须都正确。

    ⚠️ 用同一个函数服务枚举与模型两条路径，正是为了让这两种写法**只有
    一处实现** —— 否则「无 docstring 时删掉」这条规则迟早只在一边生效。
    """
    summary = first_line_summary(doc)
    if summary:
        json_schema["description"] = summary
    else:
        json_schema.pop("description", None)
    return json_schema


__all__ = [
    "apply_description",
    "first_line_summary",
]
