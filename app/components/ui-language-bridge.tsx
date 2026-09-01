'use client';

import { useEffect } from 'react';

import { translateUiText, type UiLanguage } from '../lib/i18n';

const translatedAttributes = ['aria-label', 'aria-description', 'aria-valuetext', 'title', 'placeholder', 'data-tooltip', 'data-tooltip-title'] as const;
const ignoredTags = new Set(['CODE', 'PRE', 'SCRIPT', 'STYLE']);

export function UiLanguageBridge({ language }: { language: UiLanguage }) {
  useEffect(() => {
    document.documentElement.lang = language;
    document.documentElement.dataset.locale = language;
    document.title = language === 'en' ? 'LabelOne · Image Data Workspace' : 'LabelOne · 图像数据工作台';
    const description = document.querySelector<HTMLMetaElement>('meta[name="description"]');
    if (description) description.content = language === 'en' ? 'High-throughput image dataset annotation, processing, and inference workspace' : '高速大图数据集标注、处理与推理交互原型';
    const translateTextNode = (node: Text) => {
      const parent = node.parentElement;
      if (!parent || ignoredTags.has(parent.tagName) || parent.closest('[data-i18n-ignore="true"]')) return;
      const translated = translateUiText(node.data, language);
      if (translated !== node.data) node.data = translated;
    };
    const translateElement = (element: Element) => {
      if (element.closest('[data-i18n-ignore="true"]')) return;
      for (const attribute of translatedAttributes) {
        const value = element.getAttribute(attribute);
        if (!value) continue;
        const translated = translateUiText(value, language);
        if (translated !== value) element.setAttribute(attribute, translated);
      }
      for (const child of element.childNodes) {
        if (child.nodeType === Node.TEXT_NODE) translateTextNode(child as Text);
      }
    };
    const translateTree = (root: ParentNode) => {
      if (root instanceof Element) translateElement(root);
      root.querySelectorAll?.('*').forEach(translateElement);
    };
    translateTree(document.body);
    const observer = new MutationObserver((records) => {
      for (const record of records) {
        if (record.type === 'characterData') translateTextNode(record.target as Text);
        else if (record.type === 'attributes' && record.target instanceof Element) translateElement(record.target);
        else for (const node of record.addedNodes) {
          if (node.nodeType === Node.TEXT_NODE) translateTextNode(node as Text);
          else if (node instanceof Element) translateTree(node);
        }
      }
    });
    observer.observe(document.body, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: [...translatedAttributes] });
    return () => observer.disconnect();
  }, [language]);
  return null;
}
