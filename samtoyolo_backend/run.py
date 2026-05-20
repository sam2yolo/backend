from __future__ import annotations

from .runtime_env import ensure_runtime_environment

ensure_runtime_environment()


def main() -> None:
    import uvicorn

    from .config import Settings

    settings = Settings()
    uvicorn.run(
        "samtoyolo_backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
