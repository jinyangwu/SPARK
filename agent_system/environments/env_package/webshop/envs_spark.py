import torch.multiprocessing as mp
import gym  # <-- 改为 Gymnasium
import numpy as np
import sys
import os
from typing import Optional, List, Any

# -----------------------------------------------------------------------------
# Single worker process --------------------------------------------------------
# -----------------------------------------------------------------------------

def _worker(remote, seed, env_kwargs):
    """Core loop for a subprocess that hosts a *WebAgentTextEnv* instance.

    Commands sent from the main process are *(cmd, data)* tuples:

    - **'step'** *(str)* → returns ``(obs, reward, done, info)``
    - **'reset'** *(int | None)* → returns ``(obs, info)``
    - **'render'** *(str)* → returns the value of ``env.render(mode)``.
    - **'restore_to_state'** *(list[str])* → (obs, info)
    - **'available_actions'** *(None)* → returns the list
    - **'close'** → terminates the subprocess.
    """
    # Lazy import avoids CUDA initialisation issues under ``spawn``.
    # 也在这里导入 gym，使其在工作进程中明确可用
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webshop'))
    sys.path.append(project_root)
    from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)
    
    env_kwargs['seed'] = seed
    env = gym.make('WebAgentTextEnv-v0', **env_kwargs)
    
    current_seed: Optional[int] = None # <-- 新增：追踪当前 seed 用于 restore

    try:
        while True:
            cmd, data = remote.recv()

            # -----------------------------------------------------------------
            # Environment interaction commands
            # -----------------------------------------------------------------
            if cmd == 'step':
                action = data
                obs, reward, done, info = env.step(action)
                info = dict(info or {})  # make a *copy* so we can mutate safely
                info['available_actions'] = env.get_available_actions()
                info['task_score'] = reward # 存储原始 webshop 奖励
                info['observation_text'] = obs

                # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
                if done and reward == 1.0:
                    info['won'] = True
                    reward = 10.0
                else:
                    info['won'] = False
                    reward = 0

                remote.send((obs, reward, done, info))

            elif cmd == 'reset':
                idx = data
                obs, info = env.reset(session=idx)
                info = dict(info or {})
                info['available_actions'] = env.get_available_actions()
                info['won'] = False
                info['observation_text'] = obs

                remote.send((obs, info))
                
            # --- 新增：'restore_to_state' 命令处理 ---
            elif cmd == 'restore_to_state':

                replay_actions = data
                for action in replay_actions:
                    obs, reward, done, info = env.step(action)

                remote.send((obs, info))
            # -----------------------------------------

            elif cmd == 'render':
                mode_for_render = data
                rendered = env.render(mode=mode_for_render)
                remote.send(rendered)

            elif cmd == 'available_actions':
                remote.send(env.get_available_actions())

            # -----------------------------------------------------------------
            # Book‑keeping
            # -----------------------------------------------------------------
            elif cmd == 'close':
                remote.close()
                break
            
            elif cmd == 'get_goals':
                remote.send(env.server.goals)

            else:  # pragma: no cover – helps catch typos early
                raise NotImplementedError(f"Unknown command sent to worker: {cmd}")

    finally:  # Ensure the underlying environment *always* shuts down cleanly
        env.close()


# -----------------------------------------------------------------------------
# Vectorised multi‑process environment -----------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, multi‑process wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
        env_kwargs: dict = None,
    ) -> None:
        super().__init__()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train

        self._rng = np.random.RandomState(seed)

        self._env_kwargs = env_kwargs if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}

        # -------------------------- Multiprocessing setup --------------------
        self._parent_remotes: list[mp.connection.Connection] = []
        self._workers: list[mp.Process] = []

        # --- 修改：在非 Windows 上使用 'fork' 以提高效率 ---
        if sys.platform.startswith("win"):
            ctx = mp.get_context('spawn')
        else:
            ctx = mp.get_context('fork') # 'fork' 效率更高
        # -------------------------------------------------

        for i in range(self.num_processes):
            parent_remote, child_remote = ctx.Pipe() # <-- 修改：使用 ctx.Pipe()
            worker = ctx.Process(
                target=_worker,
                args=(child_remote, seed + (i // self.group_n), self._env_kwargs),
            )
            worker.daemon = True  # auto‑kill if the main process crashes
            worker.start()
            child_remote.close()

            self._parent_remotes.append(parent_remote)
            self._workers.append(worker)

        # --- 新增：用于追踪状态的属性 ---
        self.prev_available_actions = [{} for _ in range(self.num_processes)]
        # --------------------------------
        goals_remote = self._parent_remotes[0]
        goals_remote.send(('get_goals', None))
        goals = goals_remote.recv()

        ### eval first 500
        if not self.is_train:
            self.goal_idxs = range(500)
        else:
            self.goal_idxs = range(500, len(goals))
            
        print(self.goal_idxs)

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        for remote, action in zip(self._parent_remotes, actions):
            remote.send(('step', action))

        obs_list, reward_list, done_list, info_list = [], [], [], []
        for i, remote in enumerate(self._parent_remotes): # <-- 新增：enumerate
            obs, reward, done, info = remote.recv()
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
            
            # --- 新增：更新追踪的动作 ---
            self.prev_available_actions[i] = info['available_actions']
            # --------------------------

        # --- 修改：返回签名以匹配 SciWorld ---
        return obs_list, None, reward_list, done_list, info_list
        # ----------------------------------

    # --- 新增：step_with_mask 方法 ---
    def step_with_mask(self, actions: list[str], active_mask: list[int]):
        """
        只对 active_mask 中指定的活动环境执行 step。
        """
        if len(actions) != len(active_mask):
            raise ValueError(
                f'The num of actions ({len(actions)}) must be equal to the num of active envs ({len(active_mask)})'
            )

        # 1. 仅向活动环境发送 'step' 命令
        for i, idx in enumerate(active_mask):
            remote = self._parent_remotes[idx]
            remote.send(('step', actions[i]))

        # 2. 仅从活动环境接收结果
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for i, idx in enumerate(active_mask):
            remote = self._parent_remotes[idx]
            obs, reward, done, info = remote.recv()
            
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
            
            # 更新特定环境的追踪动作
            self.prev_available_actions[idx] = info['available_actions']
            
        # 返回签名与 step() 一致
        return obs_list, None, reward_list, done_list, info_list
    # ----------------------------------

    def reset(self):
        idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        idx = np.repeat(idx, self.group_n).tolist()

        for remote, i in zip(self._parent_remotes, idx):
            remote.send(('reset', i))

        obs_list, info_list = [], []
        # --- 修改：添加错误处理和动作追踪 ---
        for i, remote in enumerate(self._parent_remotes):
            try:
                obs, info = remote.recv()
                obs_list.append(obs)
                info_list.append(info)
                self.prev_available_actions[i] = info['available_actions']
            except EOFError:
                print(f"\n!!! FATAL: Environment Worker #{i} FAILED before sending reset results !!!", file=sys.stderr)
                print("This worker likely crashed during initialization or a previous step.", file=sys.stderr)
                self.close()
                raise
        # --------------------------------------
        return obs_list, info_list

    # --- 新增：restore 方法 ---
    def restore(self, maps: List[dict[str, Any]]):
        """
        根据提供的 'maps' 列表恢复子进程的状态。
        maps: [{"parent_idx": int, "child_idx": int, "action_history": list[str]}]
        """
        # 1. 发送所有 'restore_to_state' 命令
        for map_info in maps:
            parent_idx = map_info['parent_idx']
            child_idx = map_info['child_idx']
            action_history = map_info['action_history']
            
            # 关键：从父节点复制最后已知的可用动作
            # 这假设父节点的状态是有效的
            self.prev_available_actions[child_idx] = self.prev_available_actions[parent_idx]
            
            # 向子进程发送 restore 命令
            remote = self._parent_remotes[child_idx]
            remote.send(('restore_to_state', action_history))

        # 2. 等待所有被 restore 的子进程返回其新状态
        # 这确保了所有环境在
        for map_info in maps:
            child_idx = map_info['child_idx']
            remote = self._parent_remotes[child_idx]
            try:
                obs, info = remote.recv()
                
            except EOFError:
                print(f"\n!!! FATAL: Environment Worker #{child_idx} FAILED during restore !!!", file=sys.stderr)
                raise
    # ----------------------------

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            self._parent_remotes[env_idx].send(('render', mode))
            return self._parent_remotes[env_idx].recv()

        for remote in self._parent_remotes:
            remote.send(('render', mode))
        return [remote.recv() for remote in self._parent_remotes]

    # --- 新增：属性以匹配 SciWorld ---
    @property
    def get_available_actions(self):
        return self.prev_available_actions

    @property
    def get_admissible_commands(self):
        # webshop 中 'admissible_commands' 与 'available_actions' 相同
        return self.prev_available_actions

    # ----------------------------------

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # --- 修改：更健壮的关闭逻辑 ---
        for remote in self._parent_remotes:
            try:
                remote.send(('close', None))
            except BrokenPipeError:
                # 如果工作进程已经崩溃，管道可能已损坏
                pass 
                
        for worker in self._workers:
            worker.join(timeout=5.0) # 添加超时
            if worker.is_alive():
                worker.terminate() # 强制终止卡住的进程
        # ---------------------------------
        self._closed = True

    # --- 新增：__del__ 方法 ---
    def __del__(self):  # noqa: D401
        try:
            self.close()
        except:
            # 在 __del__ 中静默失败
            pass
    # ---------------------------


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_kwargs: dict = None,
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )