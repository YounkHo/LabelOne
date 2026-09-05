'use client';

import type { BackendMode, RegisteredDataset } from '../lib/contracts';

type Props = {
  backendMode: BackendMode;
  recentProjects: RegisteredDataset[];
  openingProjectId: string | null;
  openingFolder: boolean;
  error: string;
  onOpenProject: () => void;
  onOpenRecent: (project: RegisteredDataset) => void;
  onOpenSettings: () => void;
  onRetryService: () => void;
};

function ProjectGlyph() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3.5 6.5h6l1.8 2h9.2v9.75a1.75 1.75 0 0 1-1.75 1.75H5.25a1.75 1.75 0 0 1-1.75-1.75V6.5Z" /><path d="M3.5 8.5h17" /></svg>;
}

function SettingsGlyph() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="3" /><path d="M19.2 14.8l1.4 1.1-2 3.4-1.7-.7a7.7 7.7 0 0 1-2.3 1.3l-.3 1.8h-4l-.3-1.8a7.7 7.7 0 0 1-2.3-1.3l-1.7.7-2-3.4 1.4-1.1a8 8 0 0 1 0-2.7L4 11l2-3.4 1.7.7A7.7 7.7 0 0 1 10 7l.3-1.8h4l.3 1.8a7.7 7.7 0 0 1 2.3 1.3l1.7-.7 2 3.4-1.4 1.1a8 8 0 0 1 0 2.7Z" /></svg>;
}

function formatRecentTime(value: string) {
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) return '最近使用';
  const delta = Date.now() - timestamp;
  const minutes = Math.max(0, Math.round(delta / 60_000));
  if (minutes < 1) return '刚刚';
  if (minutes < 60) return `${minutes} 分钟前`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} 小时前`;
  const days = Math.round(hours / 24);
  if (days < 30) return `${days} 天前`;
  return new Intl.DateTimeFormat('zh-CN', { month: 'short', day: 'numeric' }).format(timestamp);
}

export function WelcomeScreen(props: Props) {
  const {
    backendMode, recentProjects, openingProjectId, openingFolder, error,
    onOpenProject, onOpenRecent, onOpenSettings, onRetryService,
  } = props;
  const online = backendMode === 'online';
  const connecting = backendMode === 'probing';
  const busy = openingFolder || openingProjectId !== null;

  return <section className="welcome-screen" aria-labelledby="welcome-title">
    <div className="welcome-glow" aria-hidden="true" />
    <div className="welcome-layout">
      <section className="welcome-start">
        <div className="welcome-product-mark" aria-hidden="true"><span /></div>
        <p className="welcome-kicker">本地图像工作台</p>
        <h1 id="welcome-title">开始使用 LabelOne</h1>
        <p className="welcome-lede">打开一个本地图像项目，LabelOne 会递归索引图像并匹配同名 JSON。未打开项目前不会加载画布、标注或模型任务。</p>
        <div className="welcome-actions">
          <button type="button" className="welcome-primary" disabled={!online || busy} onClick={onOpenProject}><ProjectGlyph /><span><strong>{openingFolder ? '正在打开系统选择器…' : '打开项目'}</strong><small>选择一个图像数据集文件夹</small></span></button>
          <button type="button" className="welcome-secondary" onClick={onOpenSettings}><SettingsGlyph /><span><strong>打开设置</strong><small>模型、存储与快捷键</small></span></button>
        </div>
        <div className={`welcome-service ${online ? 'online' : connecting ? 'connecting' : 'offline'}`} role="status">
          <span className="welcome-service-dot" />
          <div><strong>{online ? '本地服务已连接' : connecting ? '正在连接本地服务' : '本地服务未连接'}</strong><small>{online ? '目录选择、索引与最近项目可用' : connecting ? '正在检查 127.0.0.1:8766' : '启动本地服务后即可打开真实项目'}</small></div>
          {!online && !connecting && <button type="button" onClick={onRetryService}>重新连接</button>}
        </div>
        {error && <p className="welcome-error" role="alert">{error}</p>}
      </section>

      <section className="recent-projects" aria-labelledby="recent-projects-title">
        <header><div><span>最近</span><h2 id="recent-projects-title">最近打开</h2></div><small>{recentProjects.length ? `${recentProjects.length} 个项目` : '本机记录'}</small></header>
        {recentProjects.length > 0 ? <div className="recent-project-list">
          {recentProjects.map((project) => {
            const opening = openingProjectId === project.dataset_id;
            const unavailable = project.source_available === false;
            return <button type="button" key={project.dataset_id} className={unavailable ? 'unavailable' : undefined} disabled={!online || busy} title={unavailable ? '源目录已移动、卸载或删除；点击查看详情' : undefined} onClick={() => onOpenRecent(project)}>
              <span className="recent-project-icon"><ProjectGlyph /></span>
              <span className="recent-project-copy"><strong>{project.name}</strong><small title={project.root_dir}>{project.root_dir}</small></span>
              <span className="recent-project-meta"><strong>{opening ? '正在打开…' : unavailable ? '无法打开' : `${project.summary.valid.toLocaleString()} 张`}</strong><small>{unavailable ? '源目录已移动或删除' : formatRecentTime(project.updated_at)}</small></span>
            </button>;
          })}
        </div> : <div className="recent-project-empty">
          <span><ProjectGlyph /></span>
          <strong>{connecting ? '正在读取最近项目…' : online ? '还没有最近项目' : '连接服务后显示最近项目'}</strong>
          <p>{online ? '打开一个文件夹后，它会出现在这里，下一次可以直接进入。' : '最近项目来自本地索引，不会上传到远程服务。'}</p>
        </div>}
        <footer><span>项目记录保存在本机</span><span>图像不会上传</span></footer>
      </section>
    </div>
  </section>;
}
