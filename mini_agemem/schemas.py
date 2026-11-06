from pydantic import BaseModel

class GenerateResponseSchema(BaseModel):
    """generate_response的扩展参数结构。"""

    reply: str = ""