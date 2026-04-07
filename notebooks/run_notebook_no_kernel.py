import base64
import contextlib
import io
import os
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
from IPython import display as ipy_display


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def to_display_output(obj):
    data = {"text/plain": repr(obj)}
    html_method = getattr(obj, "_repr_html_", None)
    if callable(html_method):
        try:
            html = html_method()
        except Exception:
            html = None
        if html:
            data["text/html"] = html
    return {
        "output_type": "display_data",
        "data": data,
        "metadata": {},
    }


def figure_output(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "output_type": "display_data",
        "data": {
            "image/png": encoded,
            "text/plain": "<matplotlib.figure.Figure>",
        },
        "metadata": {},
    }


def main():
    notebook_path = Path("notebooks/EDA.ipynb").resolve()
    nb = nbformat.read(notebook_path, as_version=4)

    execution_count = 1
    namespace = {"__name__": "__main__"}
    active_outputs = None

    def patched_display(*objects, **kwargs):
        del kwargs
        if active_outputs is None:
            return
        for obj in objects:
            active_outputs.append(to_display_output(obj))

    def patched_show(*args, **kwargs):
        del args, kwargs
        if active_outputs is None:
            return
        for num in plt.get_fignums():
            fig = plt.figure(num)
            active_outputs.append(figure_output(fig))
            plt.close(fig)

    ipy_display.display = patched_display
    plt.show = patched_show

    original_cwd = Path.cwd()
    os.chdir(notebook_path.parent)

    try:
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue

            outputs = []
            stdout = io.StringIO()
            stderr = io.StringIO()

            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                active_outputs = outputs
                try:
                    exec(cell.source, namespace)
                except Exception:
                    tb = traceback.format_exc()
                    outputs.append(
                        {
                            "output_type": "error",
                            "ename": tb.splitlines()[-1].split(":")[0],
                            "evalue": tb.splitlines()[-1],
                            "traceback": tb.splitlines(),
                        }
                    )
                    cell.execution_count = execution_count
                    if stdout.getvalue():
                        outputs.insert(
                            0,
                            {
                                "output_type": "stream",
                                "name": "stdout",
                                "text": stdout.getvalue(),
                            },
                        )
                    if stderr.getvalue():
                        outputs.append(
                            {
                                "output_type": "stream",
                                "name": "stderr",
                                "text": stderr.getvalue(),
                            }
                        )
                    cell.outputs = outputs
                    raise
                finally:
                    active_outputs = None

            if stdout.getvalue():
                outputs.insert(
                    0,
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": stdout.getvalue(),
                    },
                )
            if stderr.getvalue():
                outputs.append(
                    {
                        "output_type": "stream",
                        "name": "stderr",
                        "text": stderr.getvalue(),
                    }
                )

            cell.execution_count = execution_count
            cell.outputs = outputs
            execution_count += 1
    finally:
        os.chdir(original_cwd)

    nbformat.write(nb, notebook_path)


if __name__ == "__main__":
    main()
