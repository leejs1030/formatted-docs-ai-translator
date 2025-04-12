export enum IpcChannel {
  GetTranslations = 'get-translations',
  GetTranslationHistory = 'get-translation-history',
  UpdateTranslation = 'update-translation',
  DeleteTranslations = 'delete-translations',
  DeleteAllTranslations = 'delete-all-translations',
  GetLogs = 'get-logs',
  DeleteLogs = 'delete-logs',
  DeleteAllLogs = 'delete-all-logs',
  TranslateTextArray = 'translate-text-array',
  GetDbPath = 'get-db-path',
  CheckForUpdates = 'check-for-updates',
  DownloadUpdate = 'download-update',
  QuitAndInstall = 'quit-and-install',
  GetCurrentVersion = 'get-current-version',
  GetExamplePresets = 'get-example-presets',
  LoadExamplePreset = 'load-example-preset',
  CreateExamplePreset = 'create-example-preset',
  GetExamplePresetDetail = 'get-example-preset-detail',
  UpdateExamplePreset = 'update-example-preset',
  DeleteExamplePreset = 'delete-example-preset',
  ExportTranslations = 'export-translations',
  ImportTranslations = 'import-translations',

  // 통합 채널 - 파일과 문자열을 함께 처리
  ParseJson = 'parse-json',
  ApplyTranslationToJson = 'apply-translation-to-json',
  ParsePlainText = 'parse-plain-text',
  ApplyTranslationToPlainText = 'apply-translation-to-plain-text',
  ParseCsv = 'parse-csv',
  ApplyTranslationToCsv = 'apply-translation-to-csv',
  ParseSubtitle = 'parse-subtitle',
  ApplyTranslationToSubtitle = 'apply-translation-to-subtitle',
  OpenExternalUrl = 'open-external-url',
}
