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
  ApplyTranslationToJsonString = 'apply-translation-to-json-string',
  ApplyTranslationToJsonFile = 'apply-translation-to-json-file',
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
  ParseJsonFile = 'parse-json-file',
  ParseJsonString = 'parse-json-string',
  ParsePlainText = 'parse-plain-text',
  ApplyTranslationToPlainText = 'apply-translation-to-plain-text',
}
