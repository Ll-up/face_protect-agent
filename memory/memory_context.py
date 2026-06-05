import uuid

class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, dict] = {}
        self._history: dict[str, list[dict]] = {} #字典里装列表，列表里又装字典

    def create_session(self, user_id: str) -> dict: #创建会话对象
        session_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
        session = {  # 会话的元信息
            "id": session_id,
            "user_id": user_id,
            "created_at": None,
            "last_message": None,
        }
        self._sessions[session_id] = session #把当前会话放入SessionManager中的字典_sessions中
        self._history[session_id] = [] #创建对应的session_id的会话历史
        return session

    def get_session(self, session_id: str) -> dict | None:
        return self._sessions.get(session_id)

    def update_session(self, session_id: str, updates: dict) -> None:
        if session_id in self._sessions:
            self._sessions[session_id].update(updates)

    def append_history(self, session_id: str, role: str, content: str) -> None:
        if session_id in self._history:
            self._history[session_id].append({"role": role, "content": content})

    def get_history(self, session_id: str, limit: int = 5) -> list[dict]: #限制最大读取数为最后5条
        hist = self._history.get(session_id, [])
        return hist[-limit:] if limit else hist