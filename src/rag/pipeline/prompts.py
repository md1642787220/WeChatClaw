"""提示词模板模块：生成答案时用的 system / human 模板。

把提示词文字和拼装逻辑从引擎里拆出来，单独成一个模块，
这样以后想改话术，不用动检索和生成的流程。

Author: MADENG
Reviewer: Li Rongdong
"""
from langchain_core.prompts import ChatPromptTemplate


# system 模板：角色名由 system_role 填进来（默认「知识库助手」）。
# 遵守 OpenAI 的消息规范：第一条消息的 role 是 "system"。
SYSTEM_PROMPT_TEMPLATE = """你是企业内部{role}（角色：{role}）。请仅依据【知识片段】回答员工问题，
回答需准确、简洁，语气专业且友好。

引用要求（重要）：每个知识片段前已带编号 [1]、[2]……请在回答中**对应事实所在的句末**
使用方括号标注来源编号，例如"公司年假为 10 天[1]"。可在同一句后并列多个编号，如[1][3]。
若多个片段都支持同一句话，可写 [1][2]。禁止编造来源编号。

无命中处理：当下方【知识片段】为空或不足以回答时（如当前问题与知识库无关，或只是
闲聊/自我介绍/能力询问），请**仍以{role}身份**礼貌回复：
  - 可以简要介绍你能做什么（基于企业内部知识库回答员工关于制度、流程、FAQ 等问题）；
  - 引导用户换一种更具体的问题；
  - 不要编造公司制度/数据；不要硬说"未找到"，除非用户问的就是具体业务问题；
  - 不要泄露内部敏感信息。"""


# human 消息：带上知识片段、历史和当前问题
HUMAN_PROMPT_TEMPLATE = """【知识片段】
{context}

【历史对话】
{history}

【员工问题】
{question}
"""


# 根据系统角色名拼一个提示词模板。
#
# 参数：
#     system_role: 系统角色名（比如「知识库助手」），填进 system 消息。
#
# 返回：
#     包含 system + human 两条消息的提示词模板。
#
# 注意：
#     遵守 OpenAI 消息规范：第一条 role="system"，第二条 role="user"（human）。
def build_prompt(system_role: str):
    system_message = SYSTEM_PROMPT_TEMPLATE.format(role=system_role)
    return ChatPromptTemplate.from_messages(
        [("system", system_message), ("human", HUMAN_PROMPT_TEMPLATE)]
    )
