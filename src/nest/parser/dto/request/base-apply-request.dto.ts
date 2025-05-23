import { TranslatedTextPath } from '@/types/common';
import { BaseParseOptionsDto } from '../options/base-parse-options.dto';

export class BaseApplyRequestDto<
  TExtra,
  TOptions extends BaseParseOptionsDto = BaseParseOptionsDto,
> {
  content: string;
  translatedTextPaths: TranslatedTextPath<TExtra>[];
  options: TOptions;
}
