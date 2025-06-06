import { Part } from '@google/generative-ai';
import { Injectable } from '@nestjs/common';
import { ExampleManagerService } from '../../../../translation/example/services/example-manager.service';
import { AiPromptConverterService } from '../../../common/services/ai-prompt-converter.service';

// Enum 및 Interface 정의 추가
enum PromptRole {
  SYSTEM = 'system',
  ASSISTANT = 'assistant',
  USER = 'user',
}

enum ChatBlockRole {
  SYSTEM = 'system',
  ASSISTANT = 'MODEL',
  USER = 'USER',
}

export interface IChatBlock {
  contents: IChatContent[];
  systemInstruction: string;
}

interface IChatContent {
  role: ChatBlockRole;
  parts: Part[];
}

@Injectable()
export class GeminiPromptConverterService extends AiPromptConverterService<IChatBlock> {
  constructor(protected readonly exampleManager: ExampleManagerService) {
    super();
  }

  protected isPromptSystemRole(role: string) {
    return role.startsWith(PromptRole.SYSTEM);
  }

  protected isPromptAssistantRole(role: string) {
    return role.startsWith(PromptRole.ASSISTANT);
  }

  protected isPromptUserRole(role: string) {
    return role.startsWith(PromptRole.USER);
  }

  protected parsePromptToChatBlock({
    image,
    currentPrompt,
  }: {
    image?: string; // Base64 인코딩된 이미지 데이터
    currentPrompt: string;
  }): IChatBlock {
    {
      const blocks = currentPrompt.match(/<\|role_start:(.*?)\|>(.*?)<\|role_end\|>/gs) || [];
      const tempContents: IChatContent[] = [];
      const result: IChatBlock = {
        contents: [],
        systemInstruction: '',
      };

      blocks.forEach((block) => {
        const roleMatch = block.match(/<\|role_start:(.*?)\|>/);
        const role = roleMatch ? roleMatch[1] : '';
        const cleanBlock = block.replace(/<\|role_start:.*?\|>|\n?<\|role_end\|>/g, '');
        const text = cleanBlock.trim();

        if (this.isPromptSystemRole(role)) {
          result.systemInstruction = text;
        } else if (this.isPromptAssistantRole(role)) {
          tempContents.push({
            role: ChatBlockRole.ASSISTANT,
            parts: [{ text }],
          });
        } else if (this.isPromptUserRole(role)) {
          if (image && text.includes('{{slot::image}}')) {
            const [beforeImage, afterImage] = text.split('{{slot::image}}');
            const parts: Part[] = [];
            if (beforeImage.trim()) {
              parts.push({ text: beforeImage.trim() });
            }
            parts.push({
              inlineData: {
                mimeType: 'image/jpeg', // 이미지 타입은 고정되어 있으므로 필요시 수정
                data: image,
              },
            });
            if (afterImage.trim()) {
              parts.push({ text: afterImage.trim() });
            }
            tempContents.push({
              role: ChatBlockRole.USER,
              parts: parts,
            });
          } else {
            tempContents.push({
              role: ChatBlockRole.USER,
              parts: [{ text }],
            });
          }
        }
      });

      // Merge consecutive same roles
      for (let i = 0; i < tempContents.length; i++) {
        const current = tempContents[i];
        if (i === 0 || current.role !== tempContents[i - 1].role) {
          result.contents.push({
            role: current.role,
            parts: [...current.parts],
          });
        } else {
          result.contents[result.contents.length - 1].parts.push(...current.parts);
        }
      }

      return result;
    }
  }
}
