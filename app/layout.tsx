import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'LabelOne · 图像数据工作台',
  description: '高速大图数据集标注、处理与推理交互原型',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="zh-CN"><body>{children}</body></html>;
}
