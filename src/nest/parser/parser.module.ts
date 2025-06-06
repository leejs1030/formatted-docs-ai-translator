import { Module } from '@nestjs/common';

import { ParserIpcHandler } from './parser.ipc.handler';
import { JsonParserService } from './services/json-parser.service';
import { ParserService } from './services/parser.service';
import { PlainTextParserService } from './services/plain-text-parser.service';
import { CsvParserService } from './services/csv-parser.service';
import { SubtitleParserService } from './services/subtitle-parser.service';

@Module({
  providers: [
    ParserService,
    JsonParserService,
    ParserIpcHandler,
    PlainTextParserService,
    CsvParserService,
    SubtitleParserService,
  ],
  exports: [ParserService],
})
export class ParserModule {}
