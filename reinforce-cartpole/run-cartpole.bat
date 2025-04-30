@echo on
setlocal

for %%B in (none std) do (
    start "" python main.py --lr=0.001 --gamma=0.99 --episodes=5000 --baseline=%%B pause
)

endlocal