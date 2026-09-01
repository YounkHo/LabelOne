'use client';

import { useMemo, useState, type KeyboardEvent, type RefObject } from 'react';

import type { ApplicationSettings, BackendMode, CloudAiSettings, NetworkProxySettings, PipelineOperatorInspection, PipelineRegistryResponse } from '../lib/contracts';
import type { UiLanguage } from '../lib/i18n';
import { displayShortcut, shortcutDefinitions, shortcutGroups, type ShortcutActionId, type ShortcutMap } from '../lib/keyboard-shortcuts';
import { CustomSelect } from './custom-select';

export type GlobalSettingsSection = 'models' | 'ai' | 'system' | 'operators' | 'shortcuts';
export type CloudAiDraft = Omit<CloudAiSettings, 'credential_configured' | 'credential_source'>;

type Props = {
  section: GlobalSettingsSection;
  onSectionChange: (section: GlobalSettingsSection) => void;
  onClose: () => void;
  closeButtonRef: RefObject<HTMLButtonElement | null>;
  isFullscreen: boolean;
  onToggleFullscreen: () => void;
  language: UiLanguage;
  onToggleLanguage: () => void;
  backendMode: BackendMode;
  remoteSettings: ApplicationSettings | null;
  remoteSettingsLoading: boolean;
  modelWeightsPath: string;
  onModelWeightsPathChange: (value: string) => void;
  onPickModelWeightsPath: () => void;
  onSaveModelWeightsPath: () => void;
  modelWeightsSaving: boolean;
  modelDirectoryPicking: 'weights' | null;
  modelSettingsStatus: string;
  modelDownloadSource: string;
  onModelDownloadSourceChange: (value: string) => void;
  onSaveModelDownloadSource: () => void;
  defaultInferenceProvider: string;
  onDefaultInferenceProviderChange: (value: string) => void;
  currentInferenceModelName: string;
  currentPipelineSummary: string;
  workspaceSettingsSaving: boolean;
  workspaceSettingsStatus: string;
  onSaveWorkspaceDefaults: () => void;
  cloudAiDraft: CloudAiDraft;
  onCloudAiDraftChange: (next: CloudAiDraft) => void;
  cloudAiSaving: boolean;
  cloudAiStatus: string;
  onSaveCloudAi: () => void;
  networkProxyDraft: NetworkProxySettings;
  onNetworkProxyDraftChange: (next: NetworkProxySettings) => void;
  networkProxySaving: boolean;
  networkProxyStatus: string;
  onSaveNetworkProxy: () => void;
  operatorRegistry: PipelineRegistryResponse | null;
  operatorRegistryLoading: boolean;
  operatorImporting: boolean;
  operatorStatus: string;
  operatorInspection: PipelineOperatorInspection | null;
  onRefreshOperators: () => void;
  onChooseOperatorZip: () => void;
  onConfirmOperatorImport: () => void;
  onCancelOperatorImport: () => void;
  shortcuts: ShortcutMap;
  shortcutOverrides: Partial<Record<ShortcutActionId, string>>;
  recordingShortcut: ShortcutActionId | null;
  shortcutFeedback: string;
  useMacSymbols: boolean;
  onStartRecording: (action: ShortcutActionId) => void;
  onRecordShortcut: (action: ShortcutActionId, event: KeyboardEvent<HTMLButtonElement>) => void;
  onResetShortcut: (action: ShortcutActionId) => void;
  onResetAllShortcuts: () => void;
};

function FullscreenGlyph({ active }: { active: boolean }) {
  return <svg viewBox="0 0 20 20" aria-hidden="true">
    {active
      ? <path d="M8 3v5H3M12 3v5h5M8 17v-5H3M12 17v-5h5" />
      : <path d="M7.5 3H3v4.5M12.5 3H17v4.5M7.5 17H3v-4.5M12.5 17H17v-4.5" />}
  </svg>;
}

export function GlobalSettingsPage(props: Props) {
  const {
    section, onSectionChange, onClose, closeButtonRef, isFullscreen, onToggleFullscreen, language, onToggleLanguage,
    backendMode, remoteSettings, remoteSettingsLoading, modelWeightsPath, onModelWeightsPathChange,
    onPickModelWeightsPath, onSaveModelWeightsPath, modelWeightsSaving, modelDirectoryPicking,
    modelSettingsStatus, modelDownloadSource, onModelDownloadSourceChange,
    onSaveModelDownloadSource, defaultInferenceProvider,
    onDefaultInferenceProviderChange, currentInferenceModelName, currentPipelineSummary,
    workspaceSettingsSaving, workspaceSettingsStatus, onSaveWorkspaceDefaults,
    cloudAiDraft, onCloudAiDraftChange, cloudAiSaving,
    cloudAiStatus, onSaveCloudAi, networkProxyDraft, onNetworkProxyDraftChange,
    networkProxySaving, networkProxyStatus, onSaveNetworkProxy, operatorRegistry, operatorRegistryLoading, operatorImporting,
    operatorStatus, operatorInspection, onRefreshOperators, onChooseOperatorZip,
    onConfirmOperatorImport, onCancelOperatorImport, shortcuts, shortcutOverrides, recordingShortcut,
    shortcutFeedback, useMacSymbols, onStartRecording, onRecordShortcut, onResetShortcut,
    onResetAllShortcuts,
  } = props;
  const environmentManaged = remoteSettings?.model_weights_managed_by === 'environment';
  const offline = backendMode !== 'online';
  const [operatorSearch, setOperatorSearch] = useState('');
  const [operatorSource, setOperatorSource] = useState<'all' | 'builtin' | 'opencv' | 'custom'>('all');
  const installedKinds = useMemo(() => new Set((operatorRegistry?.installed_packages ?? []).map((item) => item.kind)), [operatorRegistry?.installed_packages]);
  const visibleOperators = useMemo(() => (operatorRegistry?.operators ?? []).filter((operator) => {
    const source = operator.source ?? (installedKinds.has(operator.kind) ? 'custom' : operator.kind.startsWith('opencv.') ? 'opencv' : 'builtin');
    if (operatorSource !== 'all' && source !== operatorSource) return false;
    const query = operatorSearch.trim().toLocaleLowerCase();
    return !query || `${operator.title} ${operator.description} ${operator.kind} ${operator.version}`.toLocaleLowerCase().includes(query);
  }), [installedKinds, operatorRegistry?.operators, operatorSearch, operatorSource]);

  const trapFocus = (event: KeyboardEvent<HTMLElement>) => {
    if (event.key !== 'Tab') return;
    const focusable = Array.from(event.currentTarget.querySelectorAll<HTMLElement>('button:not(:disabled),input:not(:disabled),[tabindex="0"]'));
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable.at(-1)!;
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };

  return <section id="global-settings-page" className="global-settings-page" role="dialog" aria-modal="true" aria-labelledby="global-settings-title" onKeyDown={trapFocus}>
    <header className="global-settings-topbar">
      <div className="global-settings-brand"><span className="brand-mark" aria-hidden="true" /><span>LabelOne</span><i>/</i><strong id="global-settings-title">全局设置</strong></div>
      <div className="global-settings-actions"><button type="button" className="settings-language-button" aria-label={language === 'zh-CN' ? '切换到英文界面' : 'Switch to Chinese'} onClick={onToggleLanguage}><span lang={language === 'zh-CN' ? 'en' : 'zh-CN'}>{language === 'zh-CN' ? 'EN' : '中'}</span></button><button type="button" className={`settings-fullscreen ${isFullscreen ? 'active' : ''}`} aria-label={isFullscreen ? '退出全屏' : '进入全屏'} aria-pressed={isFullscreen} title={isFullscreen ? '退出全屏' : '全屏'} onClick={onToggleFullscreen}><FullscreenGlyph active={isFullscreen} /></button><button ref={closeButtonRef} type="button" className="settings-back" data-tooltip-title="返回工作台" data-tooltip="关闭全局设置并返回标注工作台" data-shortcut="Esc" onClick={onClose}>← 返回工作台 <kbd>Esc</kbd></button></div>
    </header>
    <div className="global-settings-layout">
      <aside className="global-settings-nav" aria-label="设置分类">
        <div><span className="eyebrow">应用设置</span><h2>设置</h2><p>应用级配置，不属于任何图片或数据集。</p></div>
        <nav><button className={section === 'models' ? 'active' : ''} aria-current={section === 'models' ? 'page' : undefined} onClick={() => onSectionChange('models')}><span>⬡</span><div><strong>模型与存储</strong><small>本地模型、下载源与运行设备</small></div></button><button className={section === 'ai' ? 'active' : ''} aria-current={section === 'ai' ? 'page' : undefined} onClick={() => onSectionChange('ai')}><span>✦</span><div><strong>AI 服务</strong><small>云端大模型与 Agent 规划</small></div></button><button className={section === 'system' ? 'active' : ''} aria-current={section === 'system' ? 'page' : undefined} onClick={() => onSectionChange('system')}><span>◉</span><div><strong>系统与网络</strong><small>代理与本地服务网络路径</small></div></button><button className={section === 'operators' ? 'active' : ''} aria-current={section === 'operators' ? 'page' : undefined} onClick={() => onSectionChange('operators')}><span>⌘</span><div><strong>算子库</strong><small>系统导入与已安装算子</small></div></button><button className={section === 'shortcuts' ? 'active' : ''} aria-current={section === 'shortcuts' ? 'page' : undefined} onClick={() => onSectionChange('shortcuts')}><span>⌨</span><div><strong>快捷键</strong><small>导航、画布和标注工具</small></div></button></nav>
        <footer><span className={backendMode === 'online' ? 'online' : 'offline'}>●</span><div><strong>{backendMode === 'online' ? '本地服务已连接' : '本地服务未连接'}</strong><small>{backendMode === 'online' ? '模型路径可读取与保存' : '快捷键仍可在本机设置'}</small></div></footer>
      </aside>

      <main className="global-settings-content">
        {section === 'models' && <div className="settings-content-column">
          <header className="settings-page-heading"><span className="eyebrow">Models & Storage</span><h1>模型与存储</h1><p>管理模型权重位置和默认运行设备。模型选择、加载及权重导入统一在右侧推理面板完成。</p></header>

          <section className="settings-card model-directory-card">
            <header><div><strong>模型权重下载目录</strong><p>新下载的 ONNX 与模型权重将写入这里。</p></div><span className={`settings-state ${remoteSettings?.restart_required ? 'restart' : 'ready'}`}>{remoteSettingsLoading ? '读取中' : remoteSettings?.restart_required ? '等待重启' : '当前生效'}</span></header>
            <div className="settings-path-row"><input aria-label="模型权重下载目录" disabled={offline || environmentManaged || modelWeightsSaving} value={modelWeightsPath} onChange={(event) => onModelWeightsPathChange(event.target.value)} spellCheck={false} placeholder="/absolute/path/to/model-weights" /><button type="button" disabled={offline || environmentManaged || Boolean(modelDirectoryPicking)} onClick={onPickModelWeightsPath}>{modelDirectoryPicking === 'weights' ? '选择中…' : '选择文件夹'}</button><button type="button" className="settings-primary" disabled={offline || environmentManaged || modelWeightsSaving || !modelWeightsPath.trim()} onClick={onSaveModelWeightsPath}>{modelWeightsSaving ? '保存中…' : '保存路径'}</button></div>
            <div className="settings-path-meta"><span>当前生效</span><code>{remoteSettings?.effective_model_weights_dir ?? '本地服务连接后显示'}</code></div>
            {environmentManaged && <p className="settings-notice blocked">该路径由环境变量 LABELONE_MODEL_WEIGHTS_DIR 管理，界面中不可覆盖。</p>}
            {remoteSettings?.restart_required && <p className="settings-notice restart">新目录已保存，重启本地服务后生效；已有权重不会自动迁移。</p>}
            {modelSettingsStatus && <p className="settings-form-status" role="status">{modelSettingsStatus}</p>}
          </section>

          <section className="settings-card model-runtime-card">
            <header><div><strong>运行与完整性</strong><p>这些默认值会用于后续加载与下载操作。</p></div></header>
            <div className="settings-select-row"><span><strong>首选模型下载源</strong><small>支持 GitHub、ModelScope、Hugging Face；只优先模型真实注册的地址</small></span><CustomSelect ariaLabel="模型下载源" value={modelDownloadSource} options={(remoteSettings?.model_download_sources ?? [{ id: 'auto', label: '自动选择' }, { id: 'github', label: 'GitHub' }, { id: 'modelscope', label: 'ModelScope' }, { id: 'huggingface', label: 'Hugging Face' }]).map((source) => ({ value: source.id, label: source.id === 'auto' ? `${source.label} · 推荐` : source.label }))} onChange={onModelDownloadSourceChange} /></div>
            <div className="settings-select-row"><span><strong>默认运行设备</strong><small>只展示本地服务当前真实支持的执行提供器</small></span><CustomSelect ariaLabel="默认模型运行设备" value={defaultInferenceProvider} options={[{ value: 'CPUExecutionProvider', label: 'CPU · ONNX Runtime' }]} onChange={onDefaultInferenceProviderChange} /></div>
            <div className="settings-readonly-grid"><div><span>下载并发</span><strong>{remoteSettings?.model_download_concurrency ?? 1}</strong><small>避免多个大权重争抢带宽</small></div><div><span>完整性校验</span><strong>{remoteSettings?.checksum_verification === false ? '关闭' : 'SHA-256'}</strong><small>下载完成后强制计算</small></div><div><span>应用数据目录</span><code>{remoteSettings?.data_dir ?? '本地服务连接后显示'}</code><small>数据库、缓存和任务仍保留在此</small></div></div>
            <footer className="settings-action-footer"><span>{offline ? '连接本地服务后可保存下载源。' : '保存后立即用于权重列表和新的下载任务。'}</span><button type="button" className="settings-primary" disabled={offline || modelWeightsSaving} onClick={onSaveModelDownloadSource}>保存下载源</button></footer>
          </section>

          <section className="settings-card workspace-defaults-card">
            <header><div><strong>工作区默认值</strong><p>现有数据集优先恢复自己的处理流；没有专属配置的数据集使用这里的默认值。</p></div><span className="settings-state ready">全局</span></header>
            <div className="settings-readonly-grid"><div><span>当前推理模型</span><strong>{currentInferenceModelName}</strong><small>模型选择与参数会自动保存到全局设置</small></div><div><span>当前处理流</span><strong>{currentPipelineSummary}</strong><small>点击下方按钮后作为新数据集默认模板</small></div></div>
            {workspaceSettingsStatus && <p className="settings-form-status" role="status">{workspaceSettingsStatus}</p>}
            <footer className="settings-action-footer"><span>{offline ? '连接本地服务后可保存工作区默认值。' : '只更新全局默认，不覆盖已有数据集的专属配置。'}</span><button type="button" className="settings-primary" disabled={offline || workspaceSettingsSaving} onClick={onSaveWorkspaceDefaults}>{workspaceSettingsSaving ? '保存中…' : '将当前配置设为全局默认'}</button></footer>
          </section>
        </div>}

        {section === 'ai' && <div className="settings-content-column ai-settings">
          <header className="settings-page-heading"><span className="eyebrow">云端 AI</span><h1>AI 服务</h1><p>配置用于 LabelOne Agent 的云端工具规划模型。模型只能选择已有受控工具，写操作仍需人工确认。</p></header>
          <section className="settings-card ai-connection-card">
            <header><div><strong>OpenAI-compatible 服务</strong><p>配置会即时用于新的 Agent 请求。</p></div><label className="settings-inline-switch"><input type="checkbox" checked={cloudAiDraft.enabled} onChange={(event) => onCloudAiDraftChange({ ...cloudAiDraft, enabled: event.target.checked })} /><span aria-hidden="true"><i /></span><b>{cloudAiDraft.enabled ? '已启用' : '已关闭'}</b></label></header>
            <div className="ai-settings-form">
              <label><span>协议</span><CustomSelect ariaLabel="云端 AI 协议" value={cloudAiDraft.provider} options={[{ value: 'openai_compatible', label: 'OpenAI Compatible · Chat Completions' }]} onChange={(provider) => onCloudAiDraftChange({ ...cloudAiDraft, provider: provider as 'openai_compatible' })} /></label>
              <label className="wide"><span>HTTPS Endpoint</span><input value={cloudAiDraft.endpoint} onChange={(event) => onCloudAiDraftChange({ ...cloudAiDraft, endpoint: event.target.value })} placeholder="https://api.example.com/v1/chat/completions" spellCheck={false} /></label>
              <label><span>模型 ID</span><input value={cloudAiDraft.model} onChange={(event) => onCloudAiDraftChange({ ...cloudAiDraft, model: event.target.value })} placeholder="provider-model-id" spellCheck={false} /></label>
              <label><span>API Key 环境变量</span><input value={cloudAiDraft.api_key_env} onChange={(event) => onCloudAiDraftChange({ ...cloudAiDraft, api_key_env: event.target.value.toUpperCase() })} placeholder="OPENAI_API_KEY" spellCheck={false} /></label>
              <label><span>请求超时</span><div className="settings-number-input"><input type="number" min="5" max="120" value={cloudAiDraft.timeout_seconds} onChange={(event) => onCloudAiDraftChange({ ...cloudAiDraft, timeout_seconds: Number(event.target.value) })} /><b>秒</b></div></label>
              <label><span>最大输出</span><div className="settings-number-input"><input type="number" min="128" max="4096" step="128" value={cloudAiDraft.max_output_tokens} onChange={(event) => onCloudAiDraftChange({ ...cloudAiDraft, max_output_tokens: Number(event.target.value) })} /><b>tokens</b></div></label>
            </div>
            <div className={`settings-secret-notice ${remoteSettings?.cloud_ai?.credential_configured ? 'configured' : 'missing'}`}><span>{remoteSettings?.cloud_ai?.credential_configured ? '✓' : '!'}</span><div><strong>{remoteSettings?.cloud_ai?.credential_configured ? '凭据环境变量已就绪' : '尚未检测到凭据'}</strong><p>密钥只由本地服务从 <code>{cloudAiDraft.api_key_env || '环境变量'}</code> 读取；不会进入浏览器、设置 JSON、日志或 Agent 历史。</p></div></div>
            <div className="ai-data-boundary"><strong>发送边界</strong><p>云端规划器只接收你输入的 Agent 文字和可用工具清单，不发送当前图片、标注 JSON、数据集路径或工具执行结果。远程视觉推理仍保持逐次确认。</p></div>
            {cloudAiStatus && <p className="settings-form-status" role="status">{cloudAiStatus}</p>}
            <footer className="settings-action-footer"><span>{offline ? '连接本地服务后可保存云端配置。' : '配置热加载；无需重启本地服务。'}</span><button type="button" className="settings-primary" disabled={offline || cloudAiSaving} onClick={onSaveCloudAi}>{cloudAiSaving ? '保存中…' : '保存 AI 配置'}</button></footer>
          </section>
        </div>}

        {section === 'system' && <div className="settings-content-column system-settings">
          <header className="settings-page-heading"><span className="eyebrow">System & Network</span><h1>系统与网络</h1><p>控制本地服务发出的模型下载、云端 Agent 与受信远程推理请求是否经过代理。</p></header>
          <section className="settings-card network-proxy-card">
            <header><div><strong>网络代理</strong><p>只影响新的出站连接；正在执行的下载不会切换线路。</p></div><span className={`settings-state ${remoteSettings?.network_proxy_restart_required ? 'restart' : 'ready'}`}>{remoteSettingsLoading ? '读取中' : remoteSettings?.network_proxy_restart_required ? '等待重启' : '当前生效'}</span></header>
            <div className="settings-select-row"><span><strong>代理模式</strong><small>跟随启动环境、明确直连，或使用手动 HTTP(S) 代理</small></span><CustomSelect ariaLabel="网络代理模式" value={networkProxyDraft.mode} options={[{ value: 'system', label: '跟随系统 / 环境代理' }, { value: 'direct', label: '不使用代理 · 直连' }, { value: 'manual', label: '手动配置代理' }]} onChange={(mode) => onNetworkProxyDraftChange({ ...networkProxyDraft, mode: mode as NetworkProxySettings['mode'] })} /></div>
            {networkProxyDraft.mode === 'manual' && <div className="proxy-settings-form"><label><span>代理地址</span><input aria-label="手动代理地址" value={networkProxyDraft.url} onChange={(event) => onNetworkProxyDraftChange({ ...networkProxyDraft, url: event.target.value })} placeholder="http://127.0.0.1:7890" spellCheck={false} /></label><label><span>绕过代理</span><input aria-label="代理绕过地址" value={networkProxyDraft.bypass} onChange={(event) => onNetworkProxyDraftChange({ ...networkProxyDraft, bypass: event.target.value })} placeholder="localhost,127.0.0.1,::1" spellCheck={false} /></label></div>}
            <p className="settings-notice blocked">代理地址不得包含用户名或密码。需要鉴权时请在启动环境或系统代理中安全配置，不会把凭据写入设置文件。</p>
            {remoteSettings?.network_proxy_restart_required && <p className="settings-notice restart">代理设置已保存；重启本地服务后，模型下载、云端 Agent 和受信远程推理将统一使用新线路。</p>}
            {networkProxyStatus && <p className="settings-form-status" role="status">{networkProxyStatus}</p>}
            <footer className="settings-action-footer"><span>{offline ? '连接本地服务后可保存代理。' : '保存不会中断正在执行的网络任务。'}</span><button type="button" className="settings-primary" disabled={offline || networkProxySaving || (networkProxyDraft.mode === 'manual' && !networkProxyDraft.url.trim())} onClick={onSaveNetworkProxy}>{networkProxySaving ? '保存中…' : '保存代理设置'}</button></footer>
          </section>
        </div>}

        {section === 'operators' && <div className="settings-content-column operator-library-settings">
          <header className="settings-page-heading with-action"><div><span className="eyebrow">Operator Registry</span><h1>算子库</h1><p>新增算子统一从系统文件选择器逐个导入；安装成功后立即合并到处理流使用的算子注册表。</p></div><div className="settings-heading-actions"><button type="button" disabled={offline || operatorRegistryLoading} onClick={onRefreshOperators}>{operatorRegistryLoading ? '刷新中…' : '刷新'}</button><button type="button" className="settings-primary" disabled={offline || operatorImporting} onClick={onChooseOperatorZip}>{operatorImporting ? '检查中…' : '从系统导入算子'}</button></div></header>
          <div className="operator-library-toolbar"><input type="search" value={operatorSearch} onChange={(event) => setOperatorSearch(event.target.value)} placeholder="搜索名称、kind 或版本" aria-label="搜索算子库" /><CustomSelect ariaLabel="筛选算子来源" value={operatorSource} options={[{ value: 'all', label: '全部来源' }, { value: 'builtin', label: '内置' }, { value: 'opencv', label: 'OpenCV' }, { value: 'custom', label: '已导入' }]} onChange={(value) => setOperatorSource(value as typeof operatorSource)} /><span><strong>{visibleOperators.length}</strong> / {operatorRegistry?.operators.length ?? 0}</span></div>
          {operatorInspection && <section className="operator-inspection"><header><div><span className="eyebrow">导入检查通过</span><strong>{operatorInspection.operator.title}</strong><small>{operatorInspection.operator.description}</small><code>ID · {operatorInspection.operator.kind} · v{operatorInspection.operator.version}</code></div><button type="button" aria-label="取消算子安装" onClick={onCancelOperatorImport}>×</button></header><dl><div><dt>入口</dt><dd>{operatorInspection.entrypoint}</dd></div><div><dt>标注策略</dt><dd>{operatorInspection.annotation_policy}</dd></div><div><dt>SHA-256</dt><dd title={operatorInspection.digest}>{operatorInspection.digest.slice(0, 24)}…</dd></div></dl><p><strong>本地算子包：</strong>确认后会安装并探测 Python 入口；执行时使用当前用户权限，请只导入你认可的包。</p><footer><button type="button" onClick={onCancelOperatorImport}>取消</button><button type="button" className="settings-primary" disabled={operatorImporting} onClick={onConfirmOperatorImport}>{operatorImporting ? '安装并合并中…' : '安装并合并到算子库'}</button></footer></section>}
          {operatorStatus && <p className="settings-form-status" role="status">{operatorStatus}</p>}
          {(operatorRegistry?.warnings?.length ?? 0) > 0 && <section className="operator-warning-list"><strong>注册表警告</strong>{operatorRegistry!.warnings!.map((warning, index) => <p key={`${index}:${warning}`}>{warning}</p>)}</section>}
          <section className="operator-library-list" aria-label="算子列表">
            {visibleOperators.map((operator) => {
              const installed = (operatorRegistry?.installed_packages ?? []).find((item) => item.kind === operator.kind);
              const source = operator.source ?? (installed ? 'custom' : operator.kind.startsWith('opencv.') ? 'opencv' : 'builtin');
              return <article key={operator.kind}><span className={`operator-source ${source}`}>{source === 'custom' ? 'ZIP' : source === 'opencv' ? 'CV' : '内置'}</span><div><strong>{operator.title}</strong><small>{operator.description}</small><code>ID · {operator.kind}</code></div><span className="operator-contract-label">{operator.input_type} → {operator.output_type}</span><span className="operator-policy-label">标注 {String(operator.annotation_policy?.mode ?? installed?.annotation_policy ?? 'preserve')}</span><b>v{operator.version}</b></article>;
            })}
            {visibleOperators.length === 0 && <div className="operator-library-empty">没有符合筛选条件的算子。</div>}
          </section>
          <p className="operator-library-footnote">已导入包：{operatorRegistry?.installed_packages?.length ?? 0} · 注册表 <code>{operatorRegistry?.registry_hash.slice(0, 16) ?? '未连接'}</code>。导入完成后会立即刷新并出现在处理流插入菜单，无需重启。</p>
        </div>}

        {section === 'shortcuts' && <div className="settings-content-column shortcuts-settings">
          <header className="settings-page-heading with-action"><div><span className="eyebrow">Keyboard</span><h1>快捷键</h1><p>点击按键标签后直接按下新的组合键；修改会立即生效并保存在本机。</p></div><button type="button" onClick={onResetAllShortcuts}>全部恢复默认</button></header>
          {shortcutFeedback && <p className="shortcut-feedback" aria-live="polite">{shortcutFeedback}</p>}
          {shortcutGroups.map((group) => <section className="shortcut-group" key={group}><header><strong>{group}</strong><span>{shortcutDefinitions.filter((definition) => definition.group === group).length} 项</span></header><div>{shortcutDefinitions.filter((definition) => definition.group === group).map((definition) => {
            const custom = Object.hasOwn(shortcutOverrides, definition.id);
            const recording = recordingShortcut === definition.id;
            const shortcut = displayShortcut(shortcuts[definition.id], useMacSymbols);
            return <article className={`shortcut-row ${recording ? 'recording' : ''}`} key={definition.id}><div><strong>{definition.label}</strong><small>{definition.description}{definition.fixedAlternative ? ` · ${displayShortcut(definition.fixedAlternative, useMacSymbols)} 始终可用` : ''}</small></div><button type="button" className="shortcut-recorder" aria-label={`设置${definition.label}快捷键`} aria-pressed={recording} data-tooltip-title={definition.label} data-tooltip={definition.description} data-shortcut={shortcut} onClick={() => onStartRecording(definition.id)} onKeyDown={(event) => onRecordShortcut(definition.id, event)}>{recording ? <span>请按组合键…</span> : <kbd>{shortcut}</kbd>}</button><button type="button" className="shortcut-reset" disabled={!custom} aria-label={`恢复${definition.label}默认快捷键`} title="恢复默认" onClick={() => onResetShortcut(definition.id)}>↺</button></article>;
          })}</div></section>)}
          <section className="fixed-shortcuts"><strong>固定交互键</strong><p><kbd>Esc</kbd> 取消当前操作　<kbd>Enter</kbd> 完成多边形　<kbd>Space</kbd> 临时平移　<kbd>Delete</kbd> 删除选中框</p><small>这些按键属于绘制状态机和系统导航，暂不开放修改。</small></section>
        </div>}
      </main>
    </div>
  </section>;
}
