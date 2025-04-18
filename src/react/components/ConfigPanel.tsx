import HelpIcon from '@mui/icons-material/Help';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import KeyboardArrowUpIcon from '@mui/icons-material/KeyboardArrowUp';
import SaveIcon from '@mui/icons-material/Save';
import SettingsIcon from '@mui/icons-material/Settings';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  TextField,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  InputAdornment,
  IconButton,
  Button,
  Grid,
  Divider,
  Tooltip,
  Collapse,
  Paper,
  Alert,
  Snackbar,
  SelectChangeEvent,
} from '@mui/material';
import React, { useState, useEffect } from 'react';
import { TranslatorConfig } from '../../types/config';
import { Language, SourceLanguage } from '../../utils/language';
import { ConfigStore } from '../config/config-store';
import { useConfirmModal } from './common/ConfirmModal';
import { CopyButton } from './common/CopyButton';
import '../styles/ConfigPanel.css';
import {
  AiModelName,
  getDefaultModelConfig,
  getModelDescription,
  ModelConfig,
} from '../../ai/model';

// 각 모델의 기본 설정값 정의
const MODEL_DEFAULT_CONFIGS: Record<AiModelName, ModelConfig> = {
  [AiModelName.FLASH_EXP]: getDefaultModelConfig({
    modelName: AiModelName.FLASH_EXP,
  }),
  [AiModelName.FLASH_THINKING_EXP]: getDefaultModelConfig({
    modelName: AiModelName.FLASH_THINKING_EXP,
  }),
  [AiModelName.GEMINI_PRO_2_POINT_5_EXP]: getDefaultModelConfig({
    modelName: AiModelName.GEMINI_PRO_2_POINT_5_EXP,
  }),
};

// 직접 입력 모델을 위한 기본 설정값
const DEFAULT_CUSTOM_INPUT_CONFIG: ModelConfig = getDefaultModelConfig({
  modelName: AiModelName.GEMINI_PRO_2_POINT_5_EXP,
  requestsPerMinute: 25,
});

export const ConfigPanel: React.FC = () => {
  const [config, setConfig] = useState<TranslatorConfig>(() =>
    ConfigStore.getInstance().getConfig()
  );
  const [isApiKeyVisible, setIsApiKeyVisible] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [apiKeyError, setApiKeyError] = useState('');
  const [lastCustomInputConfig, setLastCustomInputConfig] = useState<ModelConfig | null>(null);
  const [isCustomInputMode, setIsCustomInputMode] = useState<boolean>(() => {
    // ConfigStore에서 isCustomInputMode 값을 가져와 사용
    return ConfigStore.getInstance().getConfig().isCustomInputMode;
  });
  const [customModelConfig, setCustomModelConfig] = useState<ModelConfig>(() => {
    // 선택된 모델에 맞는 기본 설정 사용 (customModelConfig가 있으면 그것을 우선 사용)
    return config.customModelConfig;
  });
  const { openConfirmModal } = useConfirmModal();

  useEffect(() => {
    const configStore = ConfigStore.getInstance();

    const handleConfigChange = (event: CustomEvent<TranslatorConfig>) => {
      setConfig(event.detail);
      if (event.detail.customModelConfig) {
        setCustomModelConfig(event.detail.customModelConfig);
      }
    };

    const handleConfigError = (event: CustomEvent<{ message: string }>) => {
      showSnackbar(event.detail.message);
    };

    configStore.addEventListener('configChanged', handleConfigChange);
    configStore.addEventListener('configError', handleConfigError);

    return () => {
      configStore.removeEventListener('configChanged', handleConfigChange);
      configStore.removeEventListener('configError', handleConfigError);
    };
  }, []);

  useEffect(() => {
    // 직접 입력 모드에서 필수 값이 비어있는지 체크
    if (isCustomInputMode) {
      const isValid = Boolean(
        customModelConfig.modelName &&
          customModelConfig.requestsPerMinute &&
          customModelConfig.maxOutputTokenCount
      );
      // setHasValidCustomInputs(isValid); // 이 변수는 현재 사용되지 않습니다.

      // 번역 버튼 활성화/비활성화 상태를 전역 이벤트로 발행
      window.dispatchEvent(
        new CustomEvent('configValidityChanged', {
          detail: { isValid, apiKeyError: Boolean(apiKeyError) },
        })
      );
    } else {
      // setHasValidCustomInputs(true); // 이 변수는 현재 사용되지 않습니다.
      window.dispatchEvent(
        new CustomEvent('configValidityChanged', {
          detail: { isValid: true, apiKeyError: Boolean(apiKeyError) },
        })
      );
    }
  }, [isCustomInputMode, customModelConfig, apiKeyError]);

  const handleSourceLanguageChange = (e: SelectChangeEvent<Language>) => {
    const configStore = ConfigStore.getInstance();
    configStore.updateConfig({
      sourceLanguage: e.target.value as SourceLanguage,
    });
  };

  const handleModelNameChange = (e: SelectChangeEvent<string>) => {
    const configStore = ConfigStore.getInstance();
    const selectedValue = e.target.value;

    // "직접 입력" 옵션이 선택되었는지 확인
    if (selectedValue === 'custom_input_mode') {
      setIsCustomInputMode(true);

      // 직접 입력 모드로 변경 시 마지막으로 저장된 직접 입력 설정이 있으면 복원
      if (!isCustomInputMode) {
        const newCustomConfig = lastCustomInputConfig || {
          ...DEFAULT_CUSTOM_INPUT_CONFIG,
        };
        setCustomModelConfig(newCustomConfig as ModelConfig);
        configStore.updateConfig({
          customModelConfig: newCustomConfig as ModelConfig,
          isCustomInputMode: true,
        });
      }
      return;
    }

    // 직접 입력 모드에서 다른 모델로 변경할 때 현재 설정 저장
    if (isCustomInputMode) {
      setLastCustomInputConfig(customModelConfig);
    }

    // 일반 모델 선택 모드
    setIsCustomInputMode(false);
    const newModelName = selectedValue as AiModelName;
    const newModelConfig = MODEL_DEFAULT_CONFIGS[newModelName];

    setCustomModelConfig(newModelConfig);

    // 모델이 변경되면 복사 상태 초기화
    configStore.updateConfig({
      customModelConfig: newModelConfig,
      isCustomInputMode: false,
    });
  };

  const handleCustomModelNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    // 커스텀 모델 이름 업데이트
    const newModelName = e.target.value;

    const newCustomModelConfig = {
      ...customModelConfig,
      modelName: newModelName as AiModelName,
    };

    setCustomModelConfig(newCustomModelConfig);

    const configStore = ConfigStore.getInstance();
    configStore.updateConfig({
      customModelConfig: newCustomModelConfig,
    });
  };

  const handleRequestsPerMinuteChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const inputValue = e.target.value;
    const rpm = inputValue === '' ? 0 : parseInt(inputValue, 10);

    // 빈 값이나 숫자 값 모두 허용
    const newCustomModelConfig = {
      ...customModelConfig,
      requestsPerMinute: rpm,
    };

    setCustomModelConfig(newCustomModelConfig);

    const configStore = ConfigStore.getInstance();
    configStore.updateConfig({
      customModelConfig: newCustomModelConfig,
    });
  };

  const handleMaxOutputTokensChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const inputValue = e.target.value;
    const tokens = inputValue === '' ? 0 : parseInt(inputValue, 10);

    // 빈 값이나 숫자 값 모두 허용
    const newCustomModelConfig = {
      ...customModelConfig,
      maxOutputTokenCount: tokens,
    };

    setCustomModelConfig(newCustomModelConfig);

    const configStore = ConfigStore.getInstance();
    configStore.updateConfig({
      customModelConfig: newCustomModelConfig,
    });
  };

  const handleApiKeyChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const apiKey = e.target.value;

    // API 키 유효성 검사
    if (apiKey && apiKey.length < 3) {
      setApiKeyError('API 키가 너무 짧습니다. 유효한 키인지 확인해주세요.');
    } else {
      setApiKeyError('');
    }

    const configStore = ConfigStore.getInstance();
    configStore.updateConfig({
      apiKey,
    });
  };

  const handleSaveConfig = () => {
    if (apiKeyError) {
      showSnackbar('유효하지 않은 API 키가 입력되었습니다.');
      return;
    }

    // API 키가 비어있는 경우 확인
    if (!config.apiKey || config.apiKey.trim() === '') {
      openConfirmModal({
        message: 'API 키가 입력되지 않았습니다. API 키 없이 계속하시겠습니까?',
        variant: 'warning',
        onConfirm: () => {
          saveConfigAfterConfirmation();
        },
      });
    } else {
      saveConfigAfterConfirmation();
    }
  };

  // 확인 후 설정 저장 로직을 별도 함수로 분리
  const saveConfigAfterConfirmation = () => {
    // 모델 설정 검증
    if (isCustomInputMode) {
      if (!customModelConfig.modelName) {
        showSnackbar('모델 이름이 설정되지 않았습니다.');
        return;
      }

      if (!customModelConfig.requestsPerMinute || customModelConfig.requestsPerMinute <= 0) {
        showSnackbar('유효한 분당 요청 수(RPM)를 입력해주세요.');
        return;
      }

      if (!customModelConfig.maxOutputTokenCount || customModelConfig.maxOutputTokenCount <= 0) {
        showSnackbar('유효한 최대 출력 토큰 수를 입력해주세요.');
        return;
      }
    }

    // 실제 API 호출에 사용할 설정 생성
    const savedModelConfig = {
      ...customModelConfig,
    };

    // 현재 설정을 모두 localStorage에 저장
    const configStore = ConfigStore.getInstance();

    // 이미 변경된 설정을 다시 저장하여 모든 설정이 localStorage에 반영되도록 함
    configStore.updateConfig({
      sourceLanguage: config.sourceLanguage,
      apiKey: config.apiKey,
      customModelConfig: savedModelConfig,
      isCustomInputMode: isCustomInputMode,
    });

    // 성공 메시지 표시
    showSnackbar('설정이 저장되었으며 브라우저 localStorage에 유지됩니다.');

    // API 키 미입력 경우 추가 안내
    if (!config.apiKey || config.apiKey.trim() === '') {
      setTimeout(() => {
        showSnackbar('알림: API 키 없이는 번역 기능이 작동하지 않습니다.');
      }, 3000);
    }
  };

  const toggleApiKeyVisibility = () => {
    setIsApiKeyVisible(!isApiKeyVisible);
  };

  const toggleExpanded = () => {
    setExpanded(!expanded);
  };

  const showSnackbar = (message: string) => {
    setSnackbarMessage(message);
    setSnackbarOpen(true);
  };

  const handleCloseSnackbar = () => {
    setSnackbarOpen(false);
  };

  return (
    <Card variant="outlined">
      <CardHeader
        title={
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <SettingsIcon sx={{ mr: 1, color: 'primary.main' }} />
            <Typography variant="h6" fontWeight="medium">
              번역 설정
            </Typography>
          </Box>
        }
        action={
          <Button
            onClick={toggleExpanded}
            endIcon={expanded ? <KeyboardArrowUpIcon /> : <KeyboardArrowDownIcon />}
            sx={{ textTransform: 'none' }}
          >
            {expanded ? '접기' : '설정 더보기'}
          </Button>
        }
      />
      <Divider />
      <CardContent sx={{ pt: 2, pb: expanded ? 2 : '16px' }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <FormControl fullWidth variant="outlined">
              <InputLabel id="source-language-label">원본 언어</InputLabel>
              <Select
                labelId="source-language-label"
                id="source-language"
                value={config.sourceLanguage}
                onChange={handleSourceLanguageChange}
                label="원본 언어"
              >
                <MenuItem value={Language.ENGLISH}>영어</MenuItem>
                <MenuItem value={Language.JAPANESE}>일본어</MenuItem>
                <MenuItem value={Language.CHINESE}>중국어</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={4}>
            <FormControl fullWidth variant="outlined">
              <InputLabel id="model-name-label">AI 모델</InputLabel>
              <Select
                labelId="model-name-label"
                id="model-name"
                value={isCustomInputMode ? 'custom_input_mode' : customModelConfig.modelName}
                onChange={handleModelNameChange}
                label="AI 모델"
              >
                <MenuItem value={AiModelName.FLASH_EXP}>
                  <Box>
                    <Typography>Gemini Flash</Typography>
                    <Typography variant="caption" color="text.secondary">
                      {getModelDescription(AiModelName.FLASH_EXP)}
                    </Typography>
                  </Box>
                </MenuItem>
                <MenuItem value={AiModelName.GEMINI_PRO_2_POINT_5_EXP}>
                  <Box>
                    <Typography>Gemini Pro</Typography>
                    <Typography variant="caption" color="text.secondary">
                      {getModelDescription(AiModelName.GEMINI_PRO_2_POINT_5_EXP)}
                    </Typography>
                  </Box>
                </MenuItem>
                <MenuItem value={AiModelName.FLASH_THINKING_EXP}>
                  <Box>
                    <Typography>Gemini Flash Thinking</Typography>
                    <Typography variant="caption" color="text.secondary">
                      {getModelDescription(AiModelName.FLASH_THINKING_EXP)}
                    </Typography>
                  </Box>
                </MenuItem>
                <MenuItem value="custom_input_mode">
                  <Box>
                    <Typography>직접 입력</Typography>
                    <Typography variant="caption" color="text.secondary">
                      모델명과 분당 요청 수(RPM)를 직접 지정
                    </Typography>
                  </Box>
                </MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              id="api-key"
              label="API 키"
              variant="outlined"
              type={isApiKeyVisible ? 'text' : 'password'}
              value={config.apiKey}
              onChange={handleApiKeyChange}
              error={!!apiKeyError}
              helperText={apiKeyError}
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">
                    <Tooltip title={isApiKeyVisible ? '숨기기' : '표시'}>
                      <IconButton
                        onClick={toggleApiKeyVisibility}
                        edge="end"
                        aria-label={isApiKeyVisible ? '비밀번호 숨기기' : '비밀번호 표시'}
                      >
                        {isApiKeyVisible ? <VisibilityOffIcon /> : <VisibilityIcon />}
                      </IconButton>
                    </Tooltip>
                  </InputAdornment>
                ),
              }}
            />
          </Grid>

          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              id="custom-model-name"
              label="모델 이름"
              variant="outlined"
              value={customModelConfig.modelName}
              onChange={handleCustomModelNameChange}
              helperText="Gemini 모델 ID"
              disabled={!isCustomInputMode}
              required
            />
          </Grid>

          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              id="requests-per-minute"
              label="분당 요청 수(RPM)"
              variant="outlined"
              type="number"
              value={customModelConfig.requestsPerMinute || ''}
              onChange={handleRequestsPerMinuteChange}
              InputProps={{
                inputProps: { min: 0 },
              }}
              helperText={
                isCustomInputMode && !customModelConfig.requestsPerMinute
                  ? '이 필드는 번역 실행 시 필수입니다'
                  : 'API 속도 제한에 맞는 분당 요청 수'
              }
              error={isCustomInputMode && !customModelConfig.requestsPerMinute}
              disabled={!isCustomInputMode}
              required
            />
          </Grid>

          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              id="max-output-tokens"
              label="최대 출력 토큰 수"
              variant="outlined"
              type="number"
              value={customModelConfig.maxOutputTokenCount || ''}
              onChange={handleMaxOutputTokensChange}
              InputProps={{
                inputProps: { min: 0 },
              }}
              helperText={
                isCustomInputMode && !customModelConfig.maxOutputTokenCount
                  ? '이 필드는 번역 실행 시 필수입니다'
                  : '모델이 생성할 최대 토큰 수'
              }
              error={isCustomInputMode && !customModelConfig.maxOutputTokenCount}
              disabled={!isCustomInputMode}
              required
            />
          </Grid>
        </Grid>

        <Collapse in={expanded} timeout="auto">
          <Box sx={{ mt: 3 }}>
            <Divider sx={{ mb: 3 }} />

            <Typography variant="h6" gutterBottom fontWeight="medium">
              고급 설정
            </Typography>

            <Paper variant="outlined" sx={{ p: 2, mb: 3, bgcolor: 'background.default' }}>
              <Typography variant="subtitle1" gutterBottom fontWeight="medium">
                현재 설정 정보
              </Typography>

              <Grid container spacing={2}>
                <Grid item xs={12} sm={4}>
                  <Typography variant="body2" color="text.secondary">
                    원본 언어:
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                    <Typography variant="body1" fontWeight="medium">
                      {config.sourceLanguage}
                    </Typography>
                    <CopyButton targetValue={config.sourceLanguage} size="small" />
                  </Box>
                </Grid>

                <Grid item xs={12} sm={4}>
                  <Typography variant="body2" color="text.secondary">
                    AI 모델:
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                    <Typography variant="body1" fontWeight="medium">
                      {config.customModelConfig.modelName}
                    </Typography>
                    <CopyButton targetValue={config.customModelConfig.modelName} size="small" />
                  </Box>
                </Grid>

                <Grid item xs={12} sm={4}>
                  <Typography variant="body2" color="text.secondary">
                    입력 모드:
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                    <Typography variant="body1" fontWeight="medium">
                      {config.isCustomInputMode ? '직접 입력 모드' : '기본 모델 선택 모드'}
                    </Typography>
                  </Box>
                </Grid>

                <Grid item xs={12} sm={4}>
                  <Typography variant="body2" color="text.secondary">
                    API 키 (마스킹됨):
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                    <Typography variant="body1" fontWeight="medium">
                      {isApiKeyVisible ? config.apiKey : '••••••••••••••••'}
                    </Typography>
                    <CopyButton targetValue={config.apiKey} size="small" />
                  </Box>
                </Grid>

                <Grid item xs={12} sm={4}>
                  <Typography variant="body2" color="text.secondary">
                    모델 이름:
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                    <Typography variant="body1" fontWeight="medium">
                      {customModelConfig.modelName}
                    </Typography>
                    <CopyButton targetValue={customModelConfig.modelName} size="small" />
                  </Box>
                </Grid>

                <Grid item xs={12} sm={4}>
                  <Typography variant="body2" color="text.secondary">
                    분당 요청 수(RPM):
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                    <Typography variant="body1" fontWeight="medium">
                      {customModelConfig.requestsPerMinute}
                    </Typography>
                    <CopyButton
                      targetValue={customModelConfig.requestsPerMinute.toString()}
                      size="small"
                    />
                  </Box>
                </Grid>

                <Grid item xs={12} sm={4}>
                  <Typography variant="body2" color="text.secondary">
                    최대 출력 토큰 수:
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                    <Typography variant="body1" fontWeight="medium">
                      {customModelConfig.maxOutputTokenCount}
                    </Typography>
                    <CopyButton
                      targetValue={customModelConfig.maxOutputTokenCount.toString()}
                      size="small"
                    />
                  </Box>
                </Grid>
              </Grid>

              <Box sx={{ mt: 2 }}>
                <Alert
                  severity="info"
                  icon={<HelpIcon />}
                  sx={{
                    borderRadius: '6px',
                    '& .MuiAlert-message': { display: 'flex', alignItems: 'center' },
                  }}
                >
                  <Typography variant="body2">
                    Gemini API 키는{' '}
                    <a
                      href="https://ai.google.dev/"
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{ color: 'inherit', fontWeight: 'bold' }}
                    >
                      Google AI Studio
                    </a>
                    에서 발급받을 수 있습니다.
                  </Typography>
                </Alert>
              </Box>
            </Paper>

            <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
              <Button
                variant="contained"
                color="primary"
                startIcon={<SaveIcon />}
                onClick={handleSaveConfig}
                disabled={!!apiKeyError}
              >
                설정 저장
              </Button>
            </Box>
          </Box>
        </Collapse>
      </CardContent>

      <Snackbar
        open={snackbarOpen}
        autoHideDuration={3000}
        onClose={handleCloseSnackbar}
        message={snackbarMessage}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      />
    </Card>
  );
};

export default ConfigPanel;
