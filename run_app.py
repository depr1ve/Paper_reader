"""PyCharm 入口：右键直接运行此文件即可启动 Streamlit 界面"""
import sys
import os
from streamlit.web import cli as stcli

if __name__ == "__main__":
    app_path = os.path.join(os.path.dirname(__file__), "streamlit_app.py")
    sys.argv = ["streamlit", "run", app_path, "--server.port=8501"]
    sys.exit(stcli.main())
