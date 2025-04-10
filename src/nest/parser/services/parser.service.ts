import { Injectable } from '@nestjs/common';

import { TextPath, TranslatedTextPath } from '../../../types/common';
import { JsonParserService } from './json-parser.service';
import { PlainTextParserService } from './plain-text-parser.service';
import { PlainTextParserOptionsDto } from '@/nest/parser/dto/options/plain-text-parser-options.dto';
import { JsonParserOptionsDto } from '@/nest/parser/dto/options/json-parser-options.dto';

@Injectable()
export class ParserService {
  constructor(
    private readonly jsonParserService: JsonParserService,
    private readonly plainTextParserService: PlainTextParserService
  ) {}

  public getJsonTranslationTargets(
    json: Record<string, unknown>,
    options: JsonParserOptionsDto
  ): TextPath[] {
    return this.jsonParserService.getTranslationTargets(json, options);
  }

  public applyJsonTranslation(
    json: Record<string, unknown>,
    translations: TranslatedTextPath[],
    options: JsonParserOptionsDto
  ): Record<string, unknown> {
    return this.jsonParserService.applyTranslation(json, translations, options);
  }

  public getPlainTextTranslationTargets(
    text: string,
    options: PlainTextParserOptionsDto
  ): TextPath[] {
    return this.plainTextParserService.getTranslationTargets(text, options);
  }

  public applyPlainTextTranslation(
    text: string,
    translations: TranslatedTextPath[],
    options: PlainTextParserOptionsDto
  ): string {
    return this.plainTextParserService.applyTranslation(text, translations, options);
  }
}
