import { Injectable } from '@nestjs/common';

import { SimpleTranslatedTextPath, SimpleTextPath } from '../../../types/common';
import { JsonParserService } from './json-parser.service';
import { PlainTextParserService } from './plain-text-parser.service';
import { CsvParserService } from './csv-parser.service';
import { SubtitleParserService } from './subtitle-parser.service';
import { PlainTextParserOptionsDto } from '@/nest/parser/dto/options/plain-text-parser-options.dto';
import { JsonParserOptionsDto } from '@/nest/parser/dto/options/json-parser-options.dto';
import { CsvParserOptionsDto } from '@/nest/parser/dto/options/csv-parser-options.dto';
import { SubtitleParserOptionsDto } from '@/nest/parser/dto/options/subtitle-parser-options.dto';

@Injectable()
export class ParserService {
  constructor(
    private readonly jsonParserService: JsonParserService,
    private readonly plainTextParserService: PlainTextParserService,
    private readonly csvParserService: CsvParserService,
    private readonly subtitleParserService: SubtitleParserService
  ) {}

  /**
   * JSON 파일 또는 문자열에서 번역 대상을 추출합니다.
   * @param content 파일 경로 또는 JSON 문자열
   * @param options 파싱 옵션
   */
  public async getJsonTranslationTargets(
    content: string,
    options: JsonParserOptionsDto
  ): Promise<SimpleTextPath[]> {
    return await this.jsonParserService.getTranslationTargets({
      source: content,
      options,
    });
  }

  /**
   * JSON 파일 또는 문자열에 번역을 적용합니다.
   * @param content 파일 경로 또는 JSON 문자열
   * @param translations 번역된 텍스트 목록
   * @param options 번역 적용 옵션
   */
  public async applyJsonTranslation(
    content: string,
    translations: SimpleTranslatedTextPath[],
    options: JsonParserOptionsDto
  ): Promise<Record<string, unknown>> {
    return await this.jsonParserService.applyTranslation({
      source: content,
      translations,
      options,
    });
  }

  /**
   * 일반 텍스트 파일 또는 문자열에서 번역 대상을 추출합니다.
   * @param content 파일 경로 또는 텍스트 문자열
   * @param options 파싱 옵션
   */
  public async getPlainTextTranslationTargets(
    content: string,
    options: PlainTextParserOptionsDto
  ): Promise<SimpleTextPath[]> {
    return await this.plainTextParserService.getTranslationTargets({
      source: content,
      options,
    });
  }

  /**
   * 일반 텍스트 파일 또는 문자열에 번역을 적용합니다.
   * @param content 파일 경로 또는 텍스트 문자열
   * @param translations 번역된 텍스트 목록
   * @param options 번역 적용 옵션
   */
  public async applyPlainTextTranslation(
    content: string,
    translations: SimpleTranslatedTextPath[],
    options: PlainTextParserOptionsDto
  ): Promise<string> {
    return await this.plainTextParserService.applyTranslation({
      source: content,
      translations,
      options,
    });
  }

  /**
   * CSV 파일 또는 문자열에서 번역 대상을 추출합니다.
   * @param content 파일 경로 또는 CSV 문자열
   * @param options 파싱 옵션
   */
  public async getCsvTranslationTargets(
    content: string,
    options: CsvParserOptionsDto
  ): Promise<SimpleTextPath[]> {
    return await this.csvParserService.getTranslationTargets({
      source: content,
      options,
    });
  }

  /**
   * CSV 파일 또는 문자열에 번역을 적용합니다.
   * @param content 파일 경로 또는 CSV 문자열
   * @param translations 번역된 텍스트 목록
   * @param options 번역 적용 옵션
   */
  public async applyCsvTranslation(
    content: string,
    translations: SimpleTranslatedTextPath[],
    options: CsvParserOptionsDto
  ): Promise<string> {
    return await this.csvParserService.applyTranslation({
      source: content,
      translations,
      options,
    });
  }

  /**
   * 자막(SRT/VTT) 파일 또는 문자열에서 번역 대상을 추출합니다.
   * @param content 파일 경로 또는 자막 문자열
   * @param options 파싱 옵션
   */
  public async getSubtitleTranslationTargets(
    content: string,
    options: SubtitleParserOptionsDto
  ): Promise<SimpleTextPath[]> {
    return await this.subtitleParserService.getTranslationTargets({
      source: content,
      options,
    });
  }

  /**
   * 자막(SRT/VTT) 파일 또는 문자열에 번역을 적용합니다.
   * @param content 파일 경로 또는 자막 문자열
   * @param translations 번역된 텍스트 목록
   * @param options 번역 적용 옵션
   */
  public async applySubtitleTranslation(
    content: string,
    translations: SimpleTranslatedTextPath[],
    options: SubtitleParserOptionsDto
  ): Promise<string> {
    return await this.subtitleParserService.applyTranslation({
      source: content,
      translations,
      options,
    });
  }
}
