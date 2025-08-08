
from jinja2 import Environment, FileSystemLoader, select_autoescape

def render_report(template_dir: str, context: dict) -> str:
    env = Environment(loader=FileSystemLoader(template_dir), autoescape=select_autoescape())
    tpl = env.get_template("report_template.md")
    return tpl.render(**context)
