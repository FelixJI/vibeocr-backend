# Changelog

## 0.13.3

### Bug Fixes

- **release:** 规范运行时安装器包文件名 (#66) (5940141)

## 0.13.2

### Bug Fixes

- **runtime:** 恢复模型源浅层选择 (#64) (79dd972)
- **runtime:** 交还上游模型管理 (#63) (ad82b25)

## 0.13.1

### Bug Fixes

- **runtime:** 闭合模型资产与离线发布门禁 (#58) (9a3ba62)

## 0.13.0

### Features

- **runtime:** 接入 model registry 与持久化模型 acquisition (#56) (d816a1a)

## 0.12.3

### Bug Fixes

- **runtime:** 漂移投影按覆盖 profile 取声明版本与分布绑定 (#54) (9bffd70)

## 0.12.2

### Bug Fixes

- **runtime:** base-only 安装的漂移探测改用覆盖 profile 绑定 (#52) (eca677b)

## 0.12.1

### Features

- **runtime:** 完善协议 2.7 选择与可信镜像链路 (#47) (c13e3dc)
- **runtime:** 接入 Protocol 2.7.0 组件与下载源选择契约 (#45) (4ce02d3)
- **ocr:** 接入通用 OCR 引擎选择协议与三引擎适配器 (#43) (451e772)
- **runtime:** 实现可靠维护控制面 (#26) (a5f95db)
- **runtime:** 发布可观察的 Runtime 维护状态 (#24) (c923eab)
- **runtime:** 统一单一 Runtime 与加速方案 (#10) (8c68f90)
- **ci:** 统一 CI/CD 自动化 (#11) (c7678af)
- publish VibeOCR Backend 0.7.0 (d76ce23)

### Bug Fixes

- **runtime:** 在 runtime manifest 声明三项选择 capability (#50) (6164559)
- **runtime:** 修复基础 OCR 运行时契约 (#48) (c1178ab)
- **supervisor:** 转发 MinerU 管道选项到 file_parse (#46) (ab7b71d)
- **runtime:** 修复依赖边界与失败语义 (#38) (3eebfb7)
- **runtime:** 避免 inspect 重复探测组件 (#36) (6194c06)
- **pdf:** 修复文字层偏移与字号过小 (#35) (9127533)
- **runtime:** 降低安装进度写入并恢复短暂文件锁 (#33) (bd6862c)
- **protocol:** 分离兼容范围与构建锁 (#31) (f5ad167)
- **ci:** 修复镜像标签同步并完善六仓治理 (#30) (4f1ca8c)
- **runtime:** 修复冻结 Installer 契约资源打包 (#28) (5c6c1fb)
- **release:** 统一候选派生资产归属 (#20) (a4cac60)
- **release:** 补齐 Backend 候选资产声明 (#18) (a7791cd)
- **ci:** 修复发布 tag 推送认证 (#15) (3b93e80)
- **backend:** 修复运行时并加固质量与发布链路 (#4) (3df965b)

### Performance

- **ci:** 支持统一分片门禁与取消过时 PR 运行 (#23) (b789df0)

### Dependencies

- **protocol:** 升级 Runtime Protocol 至 2.5.0 (#41) (802da7a)

## 0.12.0

### Features

- **runtime:** 完善协议 2.7 选择与可信镜像链路 (#47) (c13e3dc)
- **runtime:** 接入 Protocol 2.7.0 组件与下载源选择契约 (#45) (4ce02d3)
- **ocr:** 接入通用 OCR 引擎选择协议与三引擎适配器 (#43) (451e772)

### Bug Fixes

- **runtime:** 修复基础 OCR 运行时契约 (#48) (c1178ab)
- **supervisor:** 转发 MinerU 管道选项到 file_parse (#46) (ab7b71d)

## 0.11.2

### Bug Fixes

- **runtime:** 修复依赖边界与失败语义 (#38) (3eebfb7)

### Dependencies

- **protocol:** 升级 Runtime Protocol 至 2.5.0 (#41) (802da7a)

## 0.11.1

### Bug Fixes

- **runtime:** 避免 inspect 重复探测组件 (#36) (6194c06)
- **pdf:** 修复文字层偏移与字号过小 (#35) (9127533)
- **runtime:** 降低安装进度写入并恢复短暂文件锁 (#33) (bd6862c)

## 0.11.0

### Bug Fixes

- **protocol:** 分离兼容范围与构建锁 (#31) (f5ad167)
- **ci:** 修复镜像标签同步并完善六仓治理 (#30) (4f1ca8c)

## 0.10.1

### Bug Fixes

- **runtime:** 修复冻结 Installer 契约资源打包 (#28) (5c6c1fb)

## 0.10.0

### Features

- **runtime:** 实现可靠维护控制面 (#26) (a5f95db)

## 0.9.0

### Features

- **runtime:** 发布可观察的 Runtime 维护状态 (#24) (c923eab)

### Performance

- **ci:** 支持统一分片门禁与取消过时 PR 运行 (#23) (b789df0)

## 0.8.2

### Features

- **runtime:** 统一单一 Runtime 与加速方案 (#10) (8c68f90)
- **ci:** 统一 CI/CD 自动化 (#11) (c7678af)

### Bug Fixes

- **release:** 统一候选派生资产归属 (#20) (a4cac60)
- **release:** 补齐 Backend 候选资产声明 (#18) (a7791cd)
- **ci:** 修复发布 tag 推送认证 (#15) (3b93e80)

## 0.8.1

### Features

- **runtime:** 统一单一 Runtime 与加速方案 (#10) (8c68f90)
- **ci:** 统一 CI/CD 自动化 (#11) (c7678af)

### Bug Fixes

- **release:** 补齐 Backend 候选资产声明 (#18) (a7791cd)
- **ci:** 修复发布 tag 推送认证 (#15) (3b93e80)

## 0.8.0

### Features

- **runtime:** 统一单一 Runtime 与加速方案 (#10) (8c68f90)
- **ci:** 统一 CI/CD 自动化 (#11) (c7678af)

### Bug Fixes

- **ci:** 修复发布 tag 推送认证 (#15) (3b93e80)

## [0.7.2](https://github.com/FelixJI/vibeocr-backend/compare/v0.7.1...v0.7.2) (2026-08-02)


### Features

* publish VibeOCR Backend 0.7.0 ([d76ce23](https://github.com/FelixJI/vibeocr-backend/commit/d76ce23dc294cd29695367f8fb2548151e04e932))


### Bug Fixes

* **backend:** 修复运行时并加固质量与发布链路 ([#4](https://github.com/FelixJI/vibeocr-backend/issues/4)) ([3df965b](https://github.com/FelixJI/vibeocr-backend/commit/3df965bd65919c5c140126902363fdf7eabfb8f5))


### Build and Packaging

* **deps-dev:** bump pytest from 9.0.2 to 9.0.3 ([#5](https://github.com/FelixJI/vibeocr-backend/issues/5)) ([358d2a9](https://github.com/FelixJI/vibeocr-backend/commit/358d2a9ce1c23bc4b69ebe61d93a41a3621b90eb))

## [0.7.1](https://github.com/FelixJI/vibeocr-backend/compare/v0.7.0...v0.7.1) (2026-08-02)


### Bug Fixes

* **backend:** 修复运行时并加固质量与发布链路 ([#4](https://github.com/FelixJI/vibeocr-backend/issues/4)) ([3df965b](https://github.com/FelixJI/vibeocr-backend/commit/3df965bd65919c5c140126902363fdf7eabfb8f5))


### Build and Packaging

* **deps-dev:** bump pytest from 9.0.2 to 9.0.3 ([#5](https://github.com/FelixJI/vibeocr-backend/issues/5)) ([358d2a9](https://github.com/FelixJI/vibeocr-backend/commit/358d2a9))

All notable changes to this project will be documented in this file.
