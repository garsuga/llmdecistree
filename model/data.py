from typing import Any, Union
from pydantic import BaseModel, Field


class Node:
    """Class representing a node in the tree"""
    
    breadcrumb_name: Union[str, None] = None
    """The name of the breadcrumb found in the original dataset or None"""
    
    condition: Union[str, None] = None
    """The description or condition representing belonging to this node or None if this Node is the root"""
    
    extras: dict[str, any]
    """Extra information stored at this node for enrichment"""
    
    parent: Union['Node', None]
    """The parent Node or None if this Node is the root"""
    
    children: list['Node']
    """The child Nodes"""
    
    # TODO: create next condition when children are updated
    # TODO: children should be property
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        return self.parent == None
    
    def is_from_breadcrumb(self) -> bool:
        return self.breadcrumb_name != None or len(self.extras.items()) > 0
    
    def __init__(self, condition: str = None, parent: 'Node' = None, extras: dict[str, any] = {}):
        self.condition = condition
        self.parent = parent
        self.extras = extras or {}
        self.children = []
        
    def add_children(self, children: list['Node']):
        self.children.extend(children)
        for c in children:
            c.parent = self


class TokenCounts(BaseModel):
    prompt: int = 0
    completion: int = 0
    total: int = 0

    def __add__(self, other: Any) -> "TokenCounts":
        if not isinstance(other, TokenCounts):
            raise TypeError(f"Unsupported operand type(s) for +: 'TokenCounts' and '{type(other).__name__}'")
        return TokenCounts(
            prompt=self.prompt + other.prompt,
            completion=self.completion + other.completion,
            total=self.total + other.total,
        )

class CategoryAnswer(BaseModel):
    categories: list[str] = Field(description="A list of categories that do not overlap. Must contain at least two categories.")
    
class CategoryChoice(BaseModel):
    category_number: int = Field(description="The number of the category chosen according to the numbers in the list.")
