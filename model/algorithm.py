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
        node.breadcrumb_name = breadcrumb
        if idx + 1 < len(breadcrumb_cols):
            node.add_children(_create_partial_tree_from_breadcrumb(sub_df, node, idx + 1, breadcrumb_cols=breadcrumb_cols, extra_cols_map=extra_cols_map))
        nodes.append(node)
    return nodes
        
    
    
def create_tree_from_breadcrumbs(df: pd.DataFrame, breadcrumb_cols: list[str], extra_cols_map: dict[str, list[str]] = None, root_node_name: str = "All Products") -> Node:
    """_Create a tree from breadcrumbs left in the dataset. Suitable for datasets with an existing heirarchy. `breadcrumb_cols` is an ordered list of columns that represent heirarchical levels. `extra_cols_map` contains extra columns to store in `Node.extras` for each breadcrumb if applicable._

    
    Args:
        df (pd.DataFrame): dataset
        breadcrumb_cols (list[str]): ordered heirarchy columns
        extra_cols_map (dict[str, list[str]], optional): list of extra columns to enrich nodes for each breadcrumb column. Defaults to None.
    """
    
    root = Node(condition=root_node_name)
    root.add_children(_create_partial_tree_from_breadcrumb(df, parent=root, idx=0, breadcrumb_cols=breadcrumb_cols, extra_cols_map=extra_cols_map))
    return root

def create_tree_from_categories(df: pd.DataFrame, condition_col: str, extra_cols: list[str], root_node_name: str = "All Products") -> Node:
    root = Node(condition=root_node_name)
    children = []
    
    for i, row in df.iterrows():
        cond = row[condition_col]
        extras = {extra_col: row[extra_col] for extra_col in extra_cols}
        node = Node(condition=cond, parent=root, extras=extras)
        node.breadcrumb_name = cond
        children.append(node)
        
    root.add_children(children)
    
    return root

def _check_branches(nodes: list[Node]) -> int:
    """Gathers information about the child counts of nodes within the given subtree.

    Args:
        nodes (list[Node]): root of the current subtree
        
    Returns:
        int: number of leaves
    """
    n_sub_branches = 0
    max_sub_branches = 0
    all_children = []
    leaves = 0
    for node in nodes:
        n_sub_branches += len(node.children)
        all_children.extend(node.children)
        max_sub_branches = max(len(node.children), max_sub_branches)
        if node.is_leaf():
            leaves += 1
    
    avg_sub_branches = n_sub_branches / len(nodes)
    
    print(f"sub_branches: {n_sub_branches}, avg: {avg_sub_branches}, max: {max_sub_branches}\n")
    print(f"leaves at this level: {leaves}")
    
    if len(all_children) > 0:
        leaves += _check_branches(all_children)
        
    return leaves

def check_tree(root: Node):
    """Gathers information above the child counts of the nodes within the given subtree

    Args:
        root (Node): root of the tree
    """
    leaves = _check_branches(nodes=root.children)
    print(f"total leaves: {leaves}")


def create_vector_store(texts: list[str], embeddings: Embeddings) -> InMemoryVectorStore:
    vectorstore = InMemoryVectorStore.from_texts(
        texts,
        embedding=embeddings,
    )
    
    return vectorstore


def sample_from_embeddings(vectorstore: InMemoryVectorStore, samples: int = 10) -> list[str]:
    """Gets representative strings from the vectorstore whose vectors relatively spread out.

    Args:
        vectorstore (InMemoryVectorStore): the vector store containing strings and their embeddings
        samples (int, optional): number of representatives to retrieve. Defaults to 10.

    Returns:
        list[str]: list of representative strings
    """
    store = vectorstore.store
    # create an index to string mapping for lookups at the end
    rev_map : dict[int, str]= {}
    embeddings = []
    # iterate and populate the embedding list and reverse map
    for idx, (k, v) in enumerate(store.items()):
        vector = v['vector']
        rev_map[idx] = v['text']
        embeddings.append(vector)
    
    # normalize embeddings for clustering
    normalized_embeddings = normalize(embeddings, norm='l2', axis=1)

    k = samples

    # sample embeddings using KMeans clustering and select representatives
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(normalized_embeddings)

    from sklearn.metrics.pairwise import cosine_similarity

    representative_indices = []
    for center in kmeans.cluster_centers_:
        sims = cosine_similarity([center], normalized_embeddings)[0]
        idx = np.argmax(sims)
        representative_indices.append(idx)

    # use reverse map to convert representative indices to their original strings
    selected_keys = [rev_map[i] for i in representative_indices]
    return selected_keys




def ask_model_category(node: Node, embeddings: Embeddings, create_llm: Callable[[], BaseChatModel], previous_cats_for_retry: list[str] = []) -> Tuple[CategoryAnswer, TokenCounts]:
    """Prompts an LLM to create new categories for the children of the given node
    
    Args:
        node (Node): the node to create new child categories for
        embeddings (Embeddings): embeddings model for sampling existing children
        create_llm (Callable[[], BaseChatModel]): callable for creating new instances of the chosen LLM
    
    Returns:
        tuple[CategoryAnswer, TokenCounts]: a tuple containing the new categories and the number of tokens expended
    """
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
    
    {previous_cats_str}

    The items in the full dataset cover the scope of: {scope}

    Provide two or more new categories to lie between the parent and children.
    These categories should not overlap and should serve to divide the existing children between new nodes to improve search performance.
    Only create categories that would have one or more item fit into them.
    The categories created should cover all items that would satisfy the previous categories.
   
    Your objective is to predict the kind of items that would be categorized using the previous categories and scope and provide a new set of categories to divide them into smaller portions.
    Do not specify catch-all categories like "Other" and "None".
    
    Do not create new categories that are ambigious with, duplicates of, or direct inverses of other categories listed above.
    """.strip()
    
    previous_cats_str = ""
    if len(previous_cats_for_retry) > 0:
        previous_cats_str = f"""
            Previous categories you created for this parent category (provide more categories for the items above):
            {'\n '.join(previous_cats_for_retry)}
        """.strip()
    
    prev_condition = "All Products"
    if node.parent is not None:
        prev_condition = node.parent.condition
    prompt = ChatPromptTemplate.from_template(template)
    prompt = prompt.format(conditions="\n ".join(selected_cats), prev_conditions="\n ".join(get_node_conditions(node)), scope="Products across all industries", prev_condition=prev_condition, previous_cats_str=previous_cats_str)

    print(f"PROMPT: \n\t{prompt}\n")

    with get_openai_callback() as cb:
        llm = create_llm().with_structured_output(CategoryAnswer)
        response = llm.invoke([HumanMessage(prompt)])
        
    return response, TokenCounts(prompt=cb.prompt_tokens, completion=cb.completion_tokens, total=cb.total_tokens)
        

    
def categorize_next(item_description: str, nodes: list[Node], create_llm: Callable[[], BaseChatModel]) -> Tuple[Optional[Node], TokenCounts]:
    """Categorizes a description into a child of the given node
    
    Args:
        item_description (str): the description to be categorized
        root (Node): root node of the current subtree
        create_llm: callable to create a new instance of the chosen LLM
    
    Returns:
        Tuple[Optional[Node], TokenCounts]: a tuple containing the chosen node or `None` if no suitable node was found and the counts of the tokens expended
    """
    conditions = [n.condition for n in nodes]
    conditions.append("None of the above")
    
    conditions = [f"{i+1}: {cond}" for i,cond in enumerate(conditions)]
    
    template = """
    With the given item and categories, determine which of the categories would contain the item.
    Choose only the number of the category which would contain the item, otherwise choose 'None of the above' from the list.
    Choose the most specific category if multiple satisfy the condition above.
    
    Categories:
    {categories}
    
    Item:
    {item}
    """.strip()
    
    prompt = ChatPromptTemplate.from_template(template)
    prompt = prompt.format(categories={"\n ".join(conditions)}, item=item_description)
    
    with get_openai_callback() as cb:
        llm = create_llm().with_structured_output(CategoryChoice)
        choice: CategoryChoice = llm.invoke([HumanMessage(prompt)])
    
    choice_node = None
    
    if choice.category_number - 1 < len(nodes):
        choice_node = nodes[choice.category_number - 1]
    
    return choice_node, TokenCounts(prompt=cb.prompt_tokens, completion=cb.completion_tokens, total=cb.total_tokens)

def optimize_tree(root: Node, max_children: int, embeddings: Embeddings, create_llm: Callable[[], BaseChatModel], progress_bars: ProgressBars = None, completed_leaves=0) -> TokenCounts:
    """Mutates the tree; Optimizes the tree by creating new nodes where necessary to divide existing node children until the `max_children` is met

    Args:
        root (Node): root node of the tree
        max_children (int): max number of children that can exist at any node
        embeddings (Embeddings): embeddings model
        create_llm (Callable[[], BaseChatModel]): callable to create a new instance of the chosen llm
        progress_bars (ProgressBars, optional): Optional progress bar wrapper to add progress information (highly recommended). Defaults to None.
        completed_leaves (int, optional): Internal value used to accumulate completed leaves. Defaults to 0.
    """
    tokens = TokenCounts()
    # if the current node is a leaf then add to the counter and continue
    if root.is_leaf():
        completed_leaves += 1
        if progress_bars is not None:
            progress_bars.update_total_progress(completed_leaves)
        return tokens
    # while the children of this node do not satsify the requirement
    previous_cats = []
    while len(root.children) > max_children:
        print(f"Working on node '{root.condition}' with {len(root.children)} subcategories.")
        if progress_bars is not None:
            progress_bars.update_status(f"<b>Status:</b> Working on node '{root.condition}' with {len(root.children)} subcategories.") 
        
        # create new categories at this level by dividing existing ones
        new_cats, token_counts = ask_model_category(root, embeddings=embeddings, create_llm=create_llm, previous_cats_for_retry=previous_cats)
        tokens += token_counts
        previous_cats.extend(new_cats.categories)
        
        if progress_bars is not None:
            progress_bars.increment_tokens(token_counts)
            
        # child the new categories to the current node and save the old children for later
        old_children = root.children
        root.children = [Node(condition=c) for c in new_cats.categories]
        uncategorized: list[Node] = []
        
        print(f"Created categories: {[c.condition for c in root.children]}")
        
        if progress_bars is not None:
            progress_bars.start_new_batch(len(old_children))
            
        # categorize the old children into the new child nodes
        n_categorized: int = 0
        for node in old_children:
            choice, token_counts = categorize_next(item_description=node.condition, nodes=[*root.children], create_llm=create_llm)
            tokens += token_counts
            n_categorized += 1
            if progress_bars is not None:
                progress_bars.update_batch_progress(n_categorized)
                progress_bars.increment_tokens(token_counts)
            # if a old child node cannot be categorized it is re-childed to the current node later
            if choice is None:
                uncategorized.append(node)
            else:
                choice.add_children([node])
                
        print(f"Rebalanced {len(old_children)-len(uncategorized)}/{len(old_children)} categories.")
                
        print(f"Failed to categorize: {[u.condition for u in uncategorized]}")
        
        for child in root.children:
            if len(child.children) == 0:
                root.children.remove(child)
        
        root.add_children(uncategorized)
    
    # optimize the subtrees starting at each of the children of current node
    for child in root.children:
        tokens += optimize_tree(root=child, max_children=max_children, create_llm=create_llm, embeddings=embeddings, progress_bars=progress_bars, completed_leaves=completed_leaves)
        
    return tokens

def _correct_parent_refs(node: Node):
    """Makes sure children of the given node have a valid parent equal to the current node

    Args:
        node (Node): root of the current subtree
    """
    for c in node.children:
        c.parent = node
        _correct_parent_refs(c)

def _clean_subtree(node: Node):
    """_Mutates the tree; removes meaningless or invalid nodes that may exist._

    Args:
        node (Node): root of the current subtree
    """
    # if the node has only a single child and is not the root then "remove" the node as it is meaningless
    if len(node.children) == 1:
        if node.parent:
            n = node.children[0]
            node.parent.children.remove(node)
            node.parent.children.append(n)
            n.parent = node.parent
            
    # if the node is a leaf and it was not created as one of the original nodes, "remove" it as it is invalid
    if node.is_leaf() and not node.is_from_breadcrumb():
        if node.parent:
            node.parent.children.remove(node)
            
    for c in node.children:
        _clean_subtree(c)
        
            

def clean_tree(root: Node):
    """_Mutates the tree; removes meaningless or invalid nodes that may exist._

    Args:
        root (Node): root node of the tree
    """
    _correct_parent_refs(root)
    _clean_subtree(root)


def _correct_parent_refs(node: Node):
    """Makes sure children of the given node have a valid parent equal to the current node

    Args:
        node (Node): root of the current subtree
    """
    for c in node.children:
        c.parent = node
        _correct_parent_refs(c)

def _clean_subtree(node: Node):
    """_Mutates the tree; removes meaningless or invalid nodes that may exist._

    Args:
        node (Node): root of the current subtree
    """
    # if the node has only a single child and is not the root then "remove" the node as it is meaningless
    if len(node.children) == 1:
        if node.parent:
            n = node.children[0]
            node.parent.children.remove(node)
            node.parent.children.append(n)
            n.parent = node.parent
            _clean_subtree(node.parent)
            return
            
    # if the node is a leaf and it was not created as one of the original nodes, "remove" it as it is invalid
    if node.is_leaf() and not node.is_from_breadcrumb():
        if node.parent:
            node.parent.children.remove(node)
            _clean_subtree(node.parent)
            return
            
    for c in node.children:
        _clean_subtree(c)
        
            

def clean_tree(root: Node):
    """_Mutates the tree; removes meaningless or invalid nodes that may exist._

    Args:
        root (Node): root node of the tree
    """
    _correct_parent_refs(root)
    _clean_subtree(root)
    _correct_parent_refs(root)
