import { isNullish } from '../../../../utils/is-nullish';
import { ExampleManagerService } from '../../../translation/example/services/example-manager.service';
import { SourceLanguage, targetLanguage } from '../../../../utils/language';

export abstract class AiPromptConverterService<ChatBlockType> {
  protected readonly exampleManager: ExampleManagerService;

  // 기존 PromptConverter 클래스의 멤버 변수 및 메서드 추가
  protected readonly DEFAULT_PREFILL =
    'I understood. I have translated all sentences without omission. I must response all senteces without aborting. Pure translation result without any extra information(only prefix included):';
  protected readonly DEFAULT_PROMPT = `<|role_start:system|>
You are translator who translate the {{source_language}} text given by user to ${targetLanguage}. You are just a translator. If it's already in ${targetLanguage}, you have to output it as it is. Keep prefix format. Response only translation text and prefix, without any extra information.
No sentence should be left untranslated, or you should not respond with a blank sentence without translating.<|role_end|>
{{example}}
<|role_start:user|>
{{content}}<|role_end|>
{{prefill}}
`;

  protected getPrompt(prompt?: string) {
    return isNullish(prompt) ? this.DEFAULT_PROMPT : prompt;
  }

  protected getPrefill(prefill?: string) {
    return isNullish(prefill) ? this.DEFAULT_PREFILL : prefill;
  }

  protected async replacePrompt({
    prompt,
    sourceLanguage,
    content,
    prefill,
  }: {
    prompt?: string;
    sourceLanguage: SourceLanguage;
    content?: string;
    prefill?: string;
  }) {
    const example = await this.exampleManager.getExample(sourceLanguage);
    let currentPrompt = this.getPrompt(prompt);
    const currentPrefill = this.getPrefill(prefill);

    currentPrompt = currentPrompt.replaceAll(
      '{{example}}',
      '{{example::source}}\n{{prefill}}\n{{example::result}}'
    );

    if (example?.source) {
      currentPrompt = currentPrompt.replaceAll(
        '{{example::source}}',
        `<|role_start:user|>\n${example?.source}<|role_end|>`
      );
    } else {
      currentPrompt = currentPrompt.replaceAll('{{example::source}}', '');
    }

    if (example?.result) {
      currentPrompt = currentPrompt.replaceAll(
        '{{example::result}}',
        `<|role_start:assistant|>\n${example?.result}<|role_end|>`
      );
    } else {
      currentPrompt = currentPrompt.replaceAll('{{example::result}}', '');
    }

    if (currentPrefill) {
      currentPrompt = currentPrompt.replaceAll(
        '{{prefill}}',
        `<|role_start:assistant|>\n${currentPrefill}<|role_end|>`
      );
    } else {
      currentPrompt = currentPrompt.replaceAll('{{prefill}}', '');
    }

    if (content) {
      currentPrompt = currentPrompt.replaceAll('{{content}}', content);
    } else {
      throw new Error('Content is required');
    }

    if (sourceLanguage) {
      currentPrompt = currentPrompt.replaceAll('{{source_language}}', sourceLanguage);
    } else {
      throw new Error('Source language is required');
    }

    return currentPrompt;
  }

  public async getChatBlock({
    content,
    image,
    sourceLanguage,
    prompt,
    prefill,
  }: {
    content?: string;
    image?: string; // Base64 인코딩된 이미지 데이터
    sourceLanguage: SourceLanguage;
    prompt?: string;
    prefill?: string;
  }): Promise<ChatBlockType> {
    const currentPrompt = await this.replacePrompt({
      prompt,
      sourceLanguage,
      content,
      prefill,
    });

    return this.parsePromptToChatBlock({
      image,
      currentPrompt,
    });
  }

  protected abstract parsePromptToChatBlock({
    image,
    currentPrompt,
  }: {
    image?: string; // Base64 인코딩된 이미지 데이터
    currentPrompt: string;
  }): ChatBlockType;
}
