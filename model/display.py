import uuid
import plotly.express as px
from model.data import Node, TokenCounts
from IPython.display import display
import plotly.graph_objects as go
from ipywidgets import IntProgress, VBox, HTML, Output
import time

def display_tree(root: Node, max_depth=3):
    labels = []
    parents = []

    def traverse(node: Node, parent_label: str = ""):
        label = str(node.condition or "Unnamed")
        labels.append(label)
        parents.append(parent_label)

        for child in node.children:
            traverse(child, parent_label=label)

    traverse(root)

    fig = px.sunburst(
        names=labels,
        parents=parents,
        maxdepth=max_depth
    )
    fig.update_layout(margin=dict(t=10, l=10, r=10, b=10))
    fig.show()

# credit to GPT 4.5 for this function
def display_lazy_tree(root: Node, max_initial_depth=2):
    out = Output()
    fig = go.FigureWidget()
    id_to_node = {}

    def build_segment(node, parent_id, depth, max_depth):
        segment = {'ids': [], 'labels': [], 'parents': []}
        node_id = str(uuid.uuid4())
        id_to_node[node_id] = node
        segment['ids'].append(node_id)
        segment['labels'].append(str(node.condition))
        segment['parents'].append(parent_id)

        if depth < max_depth:
            for child in node.children:
                child_segment = build_segment(child, node_id, depth+1, max_depth)
                segment['ids'] += child_segment['ids']
                segment['labels'] += child_segment['labels']
                segment['parents'] += child_segment['parents']

        return segment

    # Initially build limited-depth tree
    segment = build_segment(root, "", 1, max_initial_depth)

    fig.add_trace(go.Sunburst(
        ids=segment['ids'],
        labels=segment['labels'],
        parents=segment['parents'],
        branchvalues="total",
        maxdepth=max_initial_depth
    ))

    fig.update_layout(
        autosize=True,
        width=None,         # None makes the width responsive (100%)
        height=800,         # Set height explicitly (adjust as necessary)
        margin=dict(t=10, l=10, r=10, b=10)
    )

    def on_click(trace, points, state):
        with out:
            if not points.point_inds:
                return
            clicked_id = trace.ids[points.point_inds[0]]
            clicked_node = id_to_node.get(clicked_id)

            if not clicked_node:
                return

            # Check if already expanded
            already_loaded = any(p == clicked_id for p in trace.parents)
            if already_loaded:
                return

            if not clicked_node.children:
                return

            # Dynamically add one-level deeper
            new_segment = {'ids': [], 'labels': [], 'parents': []}
            for child in clicked_node.children:
                child_id = str(uuid.uuid4())
                id_to_node[child_id] = child
                new_segment['ids'].append(child_id)
                new_segment['labels'].append(str(child.condition))
                new_segment['parents'].append(clicked_id)

            # Update figure data explicitly
            with fig.batch_update():
                fig.data[0].ids += tuple(new_segment['ids'])
                fig.data[0].labels += tuple(new_segment['labels'])
                fig.data[0].parents += tuple(new_segment['parents'])
                fig.data[0].maxdepth += 1

    fig.data[0].on_click(on_click)
    display(VBox([fig, out]))

def get_node_conditions(node: Node):
    n = node
    labels = []
    while n != None:
        if n.condition is not None:
            labels.append(n.condition)
        n = n.parent
        
    return list(reversed(labels))

def format_node(node: Node):
    labels = get_node_conditions(node=node)
    return " > ".join(labels)


class ProgressBars:
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0
    
    progress_total: IntProgress = None
    progress_batch: IntProgress = None
    label_total_progress: HTML = None
    label_status: HTML = None
    label_tokens: HTML = None
    ui: VBox = None
    
    start_time: float = 0
    
    def __init__(self, n_leaves: int):
        self.progress_total = IntProgress(value=0, min=0, max=n_leaves, description='Leaves Completed:', bar_style='info', layout={'width': '90%'})
        self.label_status = HTML(value="<b>Status:</b> Starting...")
        self.label_tokens = HTML(value="Prompt: 0 | Completion: 0 | total: 0")
        self.label_total_progress = HTML(value=self._format_total_progress(n_leaves=0))
        self.ui = VBox([
            self.progress_total,
            self.label_total_progress,
            self.label_status,
            self.label_tokens
        ])
        self.start_time = time.time()

    def start_new_batch(self, n_children: int):
        self.progress_batch = IntProgress(value=0, min=0, max=n_children, description='Children Categorized:', bar_style='success', layout={'width': '90%'})
        self.ui.children = (self.progress_total, self.label_total_progress, self.progress_batch, self.label_status, self.label_tokens)
    
    def update_status(self, status: str):
        self.label_status.value = status
        
    def update_batch_progress(self, n_complete: int):
        self.update_total_progress(self.progress_total.value)
        if self.progress_batch is None:
            return
        self.progress_batch.value = n_complete
        
    def increment_tokens(self, counts: TokenCounts):
        self.completion_tokens += counts.completion
        self.prompt_tokens += counts.prompt
        self.total_tokens += counts.total
        self.label_tokens.value = f"Prompt: {self.prompt_tokens:,} | Completion: {self.completion_tokens:,} | Total: {self.total_tokens:,}"
        
    def _format_total_progress(self, n_leaves: int):
        elapsed = time.time() - self.start_time
        rate = (n_leaves) / elapsed if elapsed > 0 else 0
        eta = (self.progress_total.max - (n_leaves)) / rate if rate > 0 else 0
        def format_time(seconds):
            seconds = int(seconds)
            days, seconds = divmod(seconds, 86400)
            hours, seconds = divmod(seconds, 3600)
            minutes, seconds = divmod(seconds, 60)

            return f"{days:02}:{hours:02}:{minutes:02}:{seconds:02}"
        return f"{n_leaves}/{self.progress_total.max} | Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}"
        
    def update_total_progress(self, n_leaves: int):
        self.progress_total.value = n_leaves
        self.label_total_progress.value = self._format_total_progress(self.progress_total.value)
