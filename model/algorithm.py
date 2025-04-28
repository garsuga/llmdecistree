from model.display import ProgressBars, get_node_conditions
from model.data import CategoryAnswer, CategoryChoice, Node, TokenCounts
import pandas as pd
from langchain_core.vectorstores import InMemoryVectorStore
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from langchain_core.embeddings import Embeddings
from typing import Callable, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from langchain_community.callbacks import get_openai_callback
from langchain_core.language_models.chat_models import BaseChatModel

def _create_partial_tree_from_breadcrumb(df: pd.DataFrame, parent: Node, idx: int, breadcrumb_cols: list[str], extra_cols_map: dict[str, list[str]] = None) -> list[Node]:
    breadcrumb_col = breadcrumb_cols[idx]
    breadcrumbs = pd.unique(df[breadcrumb_col])
    nodes = []
    
    for breadcrumb in breadcrumbs:
        extras = {}
        extra_cols = extra_cols_map.get(breadcrumb_col, []) if extra_cols_map else []
        sub_df = df[df[breadcrumb_col] == breadcrumb]
        for extra_col in extra_cols:
            extras[extra_col] = pd.unique(sub_df[extra_col])
        node = Node(condition=breadcrumb, parent=parent, extras=extras)
        if idx + 1 < len(breadcrumb_cols):
            node.add_children(_create_partial_tree_from_breadcrumb(sub_df, node, idx + 1, breadcrumb_cols=breadcrumb_cols, extra_cols_map=extra_cols_map))
        nodes.append(node)
    return nodes
        
        
    
    
def create_tree_from_breadcrumbs(df: pd.DataFrame, breadcrumb_cols: list[str], extra_cols_map: dict[str, list[str]] = None) -> Node:
    """_Create a tree from breadcrumbs left in the dataset. Suitable for datasets with an existing heirarchy. `breadcrumb_cols` is an ordered list of columns that represent heirarchical levels. `extra_cols_map` contains extra columns to store in `Node.extras` for each breadcrumb if applicable._

    
    Args:
        df (pd.DataFrame): dataset
        breadcrumb_cols (list[str]): ordered heirarchy columns
        extra_cols_map (dict[str, list[str]], optional): list of extra columns to enrich nodes for each breadcrumb column. Defaults to None.
    """
    
    root = Node()
    root.add_children(_create_partial_tree_from_breadcrumb(df, parent=root, idx=0, breadcrumb_cols=breadcrumb_cols, extra_cols_map=extra_cols_map))
    return root


def _check_branches(nodes: list[Node]):
    from itertools import combinations


    n_sub_branches = 0
    max_sub_branches = 0
    all_children = []
    for node in nodes:
        n_sub_branches += len(node.children)
        all_children.extend(node.children)
        max_sub_branches = max(len(node.children), max_sub_branches)
    
    avg_sub_branches = n_sub_branches / len(nodes)
    
    print(f"sub_branches: {n_sub_branches}, avg: {avg_sub_branches}, max: {max_sub_branches}\n")
    
    if len(all_children) > 0:
        _check_branches(all_children)

def check_tree(root: Node):
    _check_branches(nodes=root.children)


def create_vector_store(texts: list[str], embeddings: Embeddings) -> InMemoryVectorStore:
    vectorstore = InMemoryVectorStore.from_texts(
        texts,
        embedding=embeddings,
    )
    
    return vectorstore


def sample_from_embeddings(vectorstore: InMemoryVectorStore, samples: int = 10) -> list[list[float]]:
    store = vectorstore.store
    rev_map = {}
    embeddings = []
    for idx, (k, v) in enumerate(store.items()):
        vector = v['vector']
        rev_map[idx] = v['text']
        embeddings.append(vector)
        
    normalized_embeddings = normalize(embeddings, norm='l2', axis=1)

    k = samples

    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(normalized_embeddings)

    from sklearn.metrics.pairwise import cosine_similarity

    representative_indices = []
    for center in kmeans.cluster_centers_:
        sims = cosine_similarity([center], normalized_embeddings)[0]
        idx = np.argmax(sims)
        representative_indices.append(idx)

    selected_keys = [rev_map[i] for i in representative_indices]
    return selected_keys




def ask_model_category(node: Node, embeddings: Embeddings, create_llm: Callable[[], BaseChatModel]) -> Tuple[CategoryAnswer, TokenCounts]:
    vectorstore = create_vector_store([n.condition for n in node.children], embeddings=embeddings)
    selected_cats = sample_from_embeddings(vectorstore=vectorstore, samples=min(15, len(node.children)))
        
    template = """
    Your job is to create a set categories that will serve as nodes in a tree.
    The new nodes will lie between the parent category below and the child categories listed.
    
    Previous categories that describe these items:
    {prev_conditions}
    
    Direct parent category that should be divided:
    {prev_condition}
    
    Sample of items that should fit into the new categories:
    {conditions}

    The items in the full dataset cover the scope of: {scope}

    Provide two or more new categories to lie between the parent and children.
    These categories should not overlap and should serve to divide the existing children between new nodes to improve search performance.
    Only create categories that would have one or more item fit into them.
    The categories created should cover all items that would satisfy the previous categories.
   
    Your objective is to predict the kind of items that would be categorized using the previous categories and scope and provide a new set of categories to divide them into smaller portions.
    Do not specify catch-all categories like "Other" and "None".
    
    Do not create new categories that are ambigious with, duplicates of, or direct inverses of other categories listed above.
    """.strip()
    prev_condition = "All"
    if node.parent is not None:
        prev_condition = node.parent.condition
    prompt = ChatPromptTemplate.from_template(template)
    prompt = prompt.format(conditions="\n* ".join(selected_cats), prev_conditions="\n* ".join(get_node_conditions(node)), scope="Products across all industries", prev_condition=prev_condition)

    with get_openai_callback() as cb:
        llm = create_llm().with_structured_output(CategoryAnswer)
        response = llm.invoke([HumanMessage(prompt)])
        
    return response, TokenCounts(prompt=cb.prompt_tokens, completion=cb.completion_tokens, total=cb.total_tokens)
        

    
def categorize_next(item_description: str, root: Node, create_llm: Callable[[], BaseChatModel]) -> Tuple[Optional[Node], TokenCounts]:
    conditions = [n.condition for n in root.children]
    conditions.append("None")
    
    conditions = [f"{i+1}: {cond}" for i,cond in enumerate(conditions)]
    
    template = """
    With the given item and categories, determine which of the categories correctly describes the item.
    Choose only the number of the category which is correct, otherwise choose None from the list.
    
    Categories:
    {categories}
    
    Item:
    {item}
    """.strip()
    
    prompt = ChatPromptTemplate.from_template(template)
    prompt = prompt.format(categories={"\n* ".join(conditions)}, item=item_description)
    
    with get_openai_callback() as cb:
        llm = create_llm().with_structured_output(CategoryChoice)
        choice: CategoryChoice = llm.invoke([HumanMessage(prompt)])
    
    choice_node = None
    
    if choice.category_number - 1 < len(root.children):
        choice_node = root.children[choice.category_number - 1]
    
    return choice_node, TokenCounts(prompt=cb.prompt_tokens, completion=cb.completion_tokens, total=cb.total_tokens)

def optimize_tree(root: Node, max_children: int, embeddings: Embeddings, create_llm: Callable[[], BaseChatModel], progress_bars: ProgressBars = None, completed_leaves=0) -> None:
    if root.is_leaf():
        completed_leaves += 1
        if progress_bars is not None:
            progress_bars.update_total_progress(completed_leaves)
    while len(root.children) > max_children:
        print(f"Working on node '{root.condition}' with {len(root.children)} subcategories.")
        if progress_bars is not None:
            progress_bars.update_status(f"<b>Status:</b> Working on node '{root.condition}' with {len(root.children)} subcategories.") 
        new_cats, token_counts = ask_model_category(root, embeddings=embeddings, create_llm=create_llm)
        if progress_bars is not None:
            progress_bars.increment_tokens(token_counts)
        old_children = root.children
        root.children = [Node(condition=c) for c in new_cats.categories]
        uncategorized: list[Node] = []
        
        print(f"Created categories: {[c.condition for c in root.children]}")
        
        if progress_bars is not None:
            progress_bars.start_new_batch(len(old_children))
        n_categorized: int = 0
        for node in old_children:
            choice, token_counts = categorize_next(item_description=node.condition, root=root, create_llm=create_llm)
            n_categorized += 1
            if progress_bars is not None:
                progress_bars.update_batch_progress(n_categorized)
                progress_bars.increment_tokens(token_counts)
            if choice is None:
                uncategorized.append(node)
            else:
                choice.add_children([node])
                
        print(f"Rebalanced {len(old_children)-len(uncategorized)}/{len(old_children)} categories.")
                
        print(f"Failed to categorize: {[u.condition for u in uncategorized]}")
        
        root.add_children(uncategorized)
    
    
    for child in root.children:
        optimize_tree(root=child, max_children=max_children, create_llm=create_llm, embeddings=embeddings, progress_bars=progress_bars, completed_leaves=completed_leaves)
