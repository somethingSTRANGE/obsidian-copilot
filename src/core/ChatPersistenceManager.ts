import { getCurrentProject } from "@/aiParams";
import { AI_SENDER, USER_SENDER } from "@/constants";
import ChainManager from "@/LLMProviders/chainManager";
import { parseReasoningBlock } from "@/LLMProviders/chainRunner/utils/AgentReasoningState";
import { logError, logInfo, logWarn } from "@/logger";
import { getSettings } from "@/settings/model";
import { ChatMessage, MessageContext } from "@/types/message";
import {
  ensureFolderExists,
  extractTextFromChunk,
  formatDateTime,
  getUtf8ByteLength,
  truncateToByteLimit,
} from "@/utils";
import {
  isInVaultCache,
  listMarkdownFiles,
  patchFrontmatter,
  readFrontmatterViaAdapter,
} from "@/utils/vaultAdapterUtils";
import { App, Notice, TFile } from "obsidian";
import { MessageRepository } from "./MessageRepository";

const SAFE_FILENAME_BYTE_LIMIT = 100;

/**
 * Escape a string for safe YAML double-quoted string value
 * Escapes backslashes and double quotes to prevent YAML parsing errors
 */
function escapeYamlString(str: string): string {
  return str.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

/**
 * ChatPersistenceManager - Handles saving and loading chat messages
 *
 * This class is responsible for:
 * - Saving chat history to markdown files in the vault
 * - Loading chat history from markdown files
 * - Managing project-aware file naming
 * - Formatting chat content for storage
 */
export class ChatPersistenceManager {
  constructor(
    private app: App,
    private messageRepo: MessageRepository,
    private chainManager?: ChainManager
  ) {}

  /**
   * Save current chat history to a markdown file
   */
  async saveChat(modelKey: string): Promise<void> {
    try {
      const messages = this.messageRepo.getDisplayMessages();
      if (messages.length === 0) {
        new Notice("No messages to save.");
        return;
      }

      const settings = getSettings();
      const chatContent = this.formatChatContent(messages);
      const firstMessageEpoch = messages[0].timestamp?.epoch || Date.now();

      // Ensure the save folder exists (supports nested paths) using utility helper.
      await ensureFolderExists(settings.defaultSaveFolder);

      // Check if a file with this epoch already exists
      const existingFile = await this.findFileByEpoch(firstMessageEpoch);
      const existingFrontmatter = existingFile
        ? this.app.metadataCache.getFileCache(existingFile)?.frontmatter
        : undefined;

      let existingTopic: string | undefined = existingFrontmatter?.topic;
      let existingLastAccessedAt: number | undefined = existingFrontmatter?.lastAccessedAt;

      // For hidden directory files, metadataCache returns null — read frontmatter via adapter
      if (existingFile && !existingFrontmatter) {
        try {
          const adapterFm = await readFrontmatterViaAdapter(this.app, existingFile.path);
          if (adapterFm) {
            if (adapterFm.topic) existingTopic = adapterFm.topic;
            if (adapterFm.lastAccessedAt) existingLastAccessedAt = Number(adapterFm.lastAccessedAt);
          }
        } catch {
          // Ignore — proceed without preserved frontmatter
        }
      }

      const preferredFileName = existingFile
        ? existingFile.path
        : this.generateFileName(messages, firstMessageEpoch, existingTopic);

      const noteContent = this.generateNoteContent(
        chatContent,
        firstMessageEpoch,
        modelKey,
        existingTopic,
        existingLastAccessedAt
      );
      let targetFile: TFile | null = existingFile;

      // Check if existingFile is a real vault file (not a synthetic object for hidden dirs)
      const existingFileIsReal =
        existingFile != null && isInVaultCache(this.app, existingFile.path);

      if (existingFile && existingFileIsReal) {
        // If the file exists in the vault cache, update via vault API
        await this.app.vault.modify(existingFile, noteContent);
        logInfo(`[ChatPersistenceManager] Updated existing chat file: ${existingFile.path}`);
      } else if (
        !isInVaultCache(this.app, preferredFileName) &&
        (await this.app.vault.adapter.exists(preferredFileName))
      ) {
        // File exists on disk but not in the vault cache.
        // This happens when the save folder is a hidden directory (path starting with '.')
        // because Obsidian's metadata cache does not index hidden paths.
        await this.app.vault.adapter.write(preferredFileName, noteContent);
        new Notice("Existing chat note found - updating it now.");
        logInfo(
          `[ChatPersistenceManager] Updated existing chat file via adapter: ${preferredFileName}`
        );
      } else {
        // File doesn't exist, create a new one
        try {
          targetFile = await this.app.vault.create(preferredFileName, noteContent);
          new Notice(`Chat saved as note: ${preferredFileName}`);
          logInfo(`[ChatPersistenceManager] Created new chat file: ${preferredFileName}`);
        } catch (error) {
          if (this.isFileAlreadyExistsError(error)) {
            const conflictFile = this.app.vault.getAbstractFileByPath(preferredFileName);
            if (conflictFile && conflictFile instanceof TFile) {
              // Read existing frontmatter to preserve lastAccessedAt and topic
              const conflictFrontmatter =
                this.app.metadataCache.getFileCache(conflictFile)?.frontmatter;
              existingTopic = conflictFrontmatter?.topic ?? existingTopic;
              const conflictLastAccessedAt = conflictFrontmatter?.lastAccessedAt;

              // Regenerate content with preserved frontmatter values
              const updatedContent = this.generateNoteContent(
                chatContent,
                firstMessageEpoch,
                modelKey,
                existingTopic,
                conflictLastAccessedAt
              );
              await this.app.vault.modify(conflictFile, updatedContent);
              targetFile = conflictFile;
              new Notice("Existing chat note found - updating it now.");
              logInfo(
                `[ChatPersistenceManager] Resolved save conflict by updating existing chat file: ${conflictFile.path}`
              );
            } else {
              // File exists on disk but not in vault cache (hidden directory)
              await this.app.vault.adapter.write(preferredFileName, noteContent);
              new Notice("Existing chat note found - updating it now.");
              logInfo(
                `[ChatPersistenceManager] Resolved save conflict via adapter: ${preferredFileName}`
              );
            }
          } else if (this.isNameTooLongError(error)) {
            // Single fallback: minimal guaranteed-to-work filename with project prefix
            const currentProject = getCurrentProject();
            const filePrefix = currentProject ? `${currentProject.id}__` : "";
            const fallbackName = `${settings.defaultSaveFolder}/${filePrefix}chat-${firstMessageEpoch}.md`;

            try {
              targetFile = await this.app.vault.create(fallbackName, noteContent);
              new Notice(`Chat saved as note: ${fallbackName}`);
              logWarn(
                `[ChatPersistenceManager] Used minimal filename due to length constraints: ${fallbackName}`
              );
            } catch (fallbackError) {
              if (this.isFileAlreadyExistsError(fallbackError)) {
                const conflictFile = this.app.vault.getAbstractFileByPath(fallbackName);
                if (conflictFile && conflictFile instanceof TFile) {
                  // Read existing frontmatter to preserve lastAccessedAt
                  const conflictFrontmatter =
                    this.app.metadataCache.getFileCache(conflictFile)?.frontmatter;
                  const conflictLastAccessedAt = conflictFrontmatter?.lastAccessedAt;
                  const conflictTopic = conflictFrontmatter?.topic;

                  // Regenerate content with preserved frontmatter values
                  const updatedContent = this.generateNoteContent(
                    chatContent,
                    firstMessageEpoch,
                    modelKey,
                    conflictTopic,
                    conflictLastAccessedAt
                  );
                  await this.app.vault.modify(conflictFile, updatedContent);
                  targetFile = conflictFile;
                  new Notice("Existing chat note found - updating it now.");
                  logInfo(
                    `[ChatPersistenceManager] Resolved fallback save conflict by updating existing chat file: ${conflictFile.path}`
                  );
                } else {
                  // File exists on disk but not in vault cache (hidden directory)
                  await this.app.vault.adapter.write(fallbackName, noteContent);
                  new Notice("Existing chat note found - updating it now.");
                  logInfo(
                    `[ChatPersistenceManager] Resolved fallback save conflict via adapter: ${fallbackName}`
                  );
                }
              } else {
                throw fallbackError;
              }
            }
          } else {
            throw error;
          }
        }
      }

      this.generateTopicAsyncIfNeeded(messages, targetFile, existingTopic);
    } catch (error) {
      logError("[ChatPersistenceManager] Error saving chat:", error);
      new Notice("Failed to save chat as note. Check console for details.");
    }
  }

  /**
   * Load chat history from a markdown file
   */
  async loadChat(file: TFile): Promise<ChatMessage[]> {
    try {
      let content: string;
      try {
        content = await this.app.vault.read(file);
      } catch {
        // Fallback for hidden directory files not indexed by Obsidian
        content = await this.app.vault.adapter.read(file.path);
      }
      const messages = this.parseChatContent(content);
      logInfo(`[ChatPersistenceManager] Loaded ${messages.length} messages from ${file.path}`);
      return messages;
    } catch (error) {
      logError("[ChatPersistenceManager] Error loading chat:", error);
      new Notice("Failed to load chat history. Check console for details.");
      return [];
    }
  }

  /**
   * Get all chat history files from the vault
   */
  async getChatHistoryFiles(): Promise<TFile[]> {
    const settings = getSettings();
    const folderFiles = await listMarkdownFiles(this.app, settings.defaultSaveFolder);
    if (folderFiles.length === 0) return [];

    // Get current project ID if in a project
    const currentProject = getCurrentProject();

    // Filter files based on project context
    return folderFiles.filter((file) => {
      if (currentProject) {
        // In project mode, only show files for this project
        return file.basename.startsWith(`${currentProject.id}__`);
      } else {
        // In non-project mode, only show files without project prefix
        return !file.basename.includes("__") || !file.basename.split("__")[0];
      }
    });
  }

  /**
   * Format messages into markdown content
   */
  formatChatContent(messages: ChatMessage[]): string {
    return messages
      .map((message) => {
        const messageText = message.message;
        const context = this.formatContext(message.context);
        const timestamp = message.timestamp ? message.timestamp.display : "Unknown time";

        if (message.sender === USER_SENDER) {
          return this.formatUserMessage(messageText, context, timestamp);
        }

        if (message.sender === AI_SENDER) {
          return this.formatAIMessage(messageText, context, timestamp);
        }

        return messageText;
      })
      .join("\n\n");
  }

  formatUserMessage(message: string, context: string, timestamp: string): string {
    const body = message + context + this.formatTimestamp(timestamp);
    return this.toCalloutBlock("copilot-user", "User Prompt", "", body);
  }

  formatAIMessage(message: string, context: string, timestamp: string): string {
    let messageText = message;

    const reasoningData = parseReasoningBlock(messageText);
    if (reasoningData) {
      messageText = reasoningData.contentAfter;
    }

    const { think, content } = this.extractThinkBlock(messageText);
    let output = "";
    if (think) {
      output += this.toCalloutBlock("copilot-ai-think", "Thought for a while", "-", think) + "\n\n";
    }

    output += content;

    if (context) {
      output += context;
    }

    output += this.formatTimestamp(timestamp);

    return output;
  }

  extractThinkBlock(message: string) {
    const match = message.match(/<think>([\s\S]*?)<\/think>/);

    if (!match) {
      return { think: null, content: message };
    }

    return {
      think: match[1].trim(),
      content: message.replace(match[0], "").trim(),
    };
  }

  formatTimestamp(timestamp: string): string {
    return `\n\n<time>${timestamp}</time>`;
  }

  formatContext(context: MessageContext | undefined): string {
    if (!context) return "";

    const hasAny =
      context.notes?.length ||
      context.urls?.length ||
      context.webTabs?.length ||
      context.tags?.length ||
      context.folders?.length;

    if (!hasAny) return "";

    const formatContextList = <T>(
      items: readonly T[] | undefined,
      mapper: (item: T) => string
    ): string => {
      return items?.length ? items.map(mapper).join("") : "";
    };

    const notes = formatContextList(context.notes, (s) => `\n>   - [[${s.path}|${s.basename}]]`);

    const urls = formatContextList(context.urls, (url) => `\n>   - ${url}`);

    const webTabs = formatContextList(
      context.webTabs,
      (s) => `\n>   - [![favicon](${s.faviconUrl}) ${s.title}](${s.url})`
    );

    const tags = formatContextList(context.tags, (tag) => `\n>   - ${tag}`);

    const folders = formatContextList(context.folders, (folder) => `\n>   - ${folder}`);

    return (
      `\n\n> [!copilot-context] Context` +
      `\n> - Notes${notes}` +
      `\n> - URLs${urls}` +
      `\n> - Web Tabs${webTabs}` +
      `\n> - Tags${tags}` +
      `\n> - Folders${folders}`
    );
  }

  toCalloutBlock(type: string, title: string, foldMarker: string, content: string): string {
    const prefixed = content
      .split("\n")
      .map((line) => `> ${line}`)
      .join("\n");

    const validMarker = foldMarker === "-" || foldMarker === "+" ? foldMarker : "";
    const titlePart = title ? ` ${title}` : "";

    return `> [!${type}]${validMarker}${titlePart}\n${prefixed}`;
  }

  async renameFileToMatchTopic(file: TFile, topic: string): Promise<void> {
    if (!file || !topic) return;

    const cache = this.app.metadataCache.getFileCache(file);
    const epoch = cache?.frontmatter?.epoch;
    if (!epoch) {
      return;
    }

    const messages = this.messageRepo.getDisplayMessages();
    const newPath = this.generateFileName(messages, epoch, topic);

    console.log("cache:", cache);
    console.log("epoch:", epoch);
    console.log("messages:", messages);
    console.log("newPath:", newPath);

    if (file.path === newPath) {
      return;
    }

    await this.app.fileManager.renameFile(file, newPath);

    return;
  }

  /**
   * Format messages into markdown content
   */
  private formatChatContent_old(messages: ChatMessage[]): string {
    return messages
      .map((message) => {
        const timestamp = message.timestamp ? message.timestamp.display : "Unknown time";

        // Strip agent reasoning block from AI messages before saving
        let messageText = message.message;
        if (message.sender === AI_SENDER) {
          const reasoningData = parseReasoningBlock(messageText);
          if (reasoningData) {
            messageText = reasoningData.contentAfter;
          }
        }

        let content = `**${message.sender}**: ${messageText}`;

        // Include context information if present
        if (message.context) {
          const contextParts: string[] = [];

          if (message.context.notes?.length) {
            contextParts.push(
              `Notes: ${message.context.notes.map((note) => note.path).join(", ")}`
            );
          }

          if (message.context.urls?.length) {
            contextParts.push(`URLs: ${message.context.urls.join(", ")}`);
          }

          if (message.context.webTabs?.length) {
            contextParts.push(
              `Web Tabs: ${message.context.webTabs.map((tab) => tab.url).join(", ")}`
            );
          }

          if (message.context.tags?.length) {
            contextParts.push(`Tags: ${message.context.tags.join(", ")}`);
          }

          if (message.context.folders?.length) {
            contextParts.push(`Folders: ${message.context.folders.join(", ")}`);
          }

          if (contextParts.length > 0) {
            content += `\n[Context: ${contextParts.join(" | ")}]`;
          }
        }

        content += `\n[Timestamp: ${timestamp}]`;
        return content;
      })
      .join("\n\n");
  }

  /**
   * Remove YAML frontmatter from the chat content
   */
  private getChatContentWithoutFrontmatter(content: string): string {
    const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---/);
    return frontmatterMatch ? content.slice(frontmatterMatch[0].length).trim() : content;
  }

  /**
   * Determine whether the chat content uses the callout-style format
   */
  private isCalloutChatFormat(content: string): boolean {
    const chatContent = this.getChatContentWithoutFrontmatter(content);
    return chatContent.includes("> [!copilot-user]");
  }

  /**
   * Parse chat content in the callout format using Obsidian callout indicators and richer Markdown
   */
  private parseCalloutChatFormat(content: string): ChatMessage[] {
    const chatContent = this.stripFrontmatter(content);
    const lines = chatContent.split("\n");
    const messages: ChatMessage[] = [];

    let remainingLines = [...lines];

    while (remainingLines.length > 0) {
      const { chatMessage, remainingLines: nextLines } = this.parseChatMessage(remainingLines);
      if (chatMessage) messages.push(chatMessage);
      remainingLines = nextLines;
    }

    return messages;
  }

  /**
   * Parse the Markdown content back into messages
   */
  private parseChatContent(content: string): ChatMessage[] {
    const chatContent = this.getChatContentWithoutFrontmatter(content);
    return this.isCalloutChatFormat(chatContent)
      ? this.parseCalloutChatFormat(chatContent)
      : this.parseInlineChatFormat(chatContent);
  }

  /**
   * Determine the type of the next message and delegate to the appropriate parser
   */
  private parseChatMessage(lines: string[]): {
    chatMessage: ChatMessage | null;
    remainingLines: string[];
  } {
    let i = 0;

    // Skip leading blank lines
    while (i < lines.length && lines[i].trim() === "") i++;
    if (i >= lines.length) return { chatMessage: null, remainingLines: [] };

    // Only pass unprocessed lines to the parser
    const unprocessedLines = lines.slice(i);
    const line = unprocessedLines[0];

    if (line.trimStart().startsWith("> [!copilot-user]")) {
      return this.parseChatMessageFromUser(unprocessedLines);
    } else {
      return this.parseChatMessageFromAI(unprocessedLines);
    }
  }

  /**
   * Parse an AI message from callout-style chat content
   */
  private parseChatMessageFromAI(lines: string[]): {
    chatMessage: ChatMessage;
    remainingLines: string[];
  } {
    const parseTimeTag = (line: string) => {
      const match = line.match(/<time>(.*?)<\/time>/);
      return match ? match[1] : null;
    };

    let i = 0;
    const aiLines: string[] = [];
    let timestampStr: string | null = null;

    while (i < lines.length) {
      const t = parseTimeTag(lines[i]);
      if (t) {
        timestampStr = t;
        i++;
        break;
      }
      aiLines.push(lines[i]);
      i++;
    }

    if (!timestampStr) timestampStr = "Unknown time";

    const messageText = aiLines.join("\n").trim();

    const timestampObj = (() => {
      const d = new Date(timestampStr!);
      return isNaN(d.getTime())
        ? null
        : { epoch: d.getTime(), display: timestampStr!, fileName: "" };
    })();

    return {
      chatMessage: {
        message: messageText,
        sender: AI_SENDER,
        isVisible: true,
        timestamp: timestampObj,
        context: undefined,
      },
      remainingLines: lines.slice(i),
    };
  }

  /**
   * Parse a user message from callout-style chat content
   */
  private parseChatMessageFromUser(lines: string[]): {
    chatMessage: ChatMessage;
    remainingLines: string[];
  } {
    const stripBlockquote = (line: string) =>
      line.startsWith(">") ? line.slice(1).replace(/^ ?/, "") : line;
    const parseTimeTag = (line: string) => {
      const match = line.match(/<time>(.*?)<\/time>/);
      return match ? match[1] : null;
    };

    let i = 0;
    const userLines: string[] = [];

    while (i < lines.length && lines[i].startsWith(">")) {
      userLines.push(lines[i]);
      i++;
    }

    // Remove header line and blank blockquote lines
    const normalized = userLines
      .slice(1)
      .map(stripBlockquote)
      .filter((l) => l !== "");

    // --- Extract timestamp (last <time> line) ---
    let timestampStr: string | null = null;
    if (normalized.length > 0) {
      const lastLine = normalized[normalized.length - 1];
      const t = parseTimeTag(lastLine);
      if (t) {
        timestampStr = t;
        normalized.pop();
      }
    }
    if (!timestampStr) timestampStr = "Unknown time";

    // --- Extract context ---
    const ctxIdx = normalized.findIndex(
      (l) => l.startsWith("[!copilot-context]") || l.startsWith("> [!copilot-context]")
    );

    let messageTextLines: string[];
    let context: any;

    if (ctxIdx >= 0) {
      messageTextLines = normalized.slice(0, ctxIdx);
      const ctxLines = normalized.slice(ctxIdx + 1);
      context = this.parseContextCallout(ctxLines);
    } else {
      messageTextLines = normalized;
    }

    const messageText = messageTextLines.join("\n").trim();

    const timestampObj = (() => {
      const d = new Date(timestampStr!);
      return isNaN(d.getTime())
        ? null
        : { epoch: d.getTime(), display: timestampStr!, fileName: "" };
    })();

    return {
      chatMessage: {
        message: messageText,
        sender: USER_SENDER,
        isVisible: true,
        timestamp: timestampObj,
        context,
      },
      remainingLines: lines.slice(i),
    };
  }

  /**
   * Parse context callout strings back into a context object
   */
  private parseContextCallout(ctxLines: string[]): any {
    if (!ctxLines.length) return undefined;

    // Remove blockquote prefix
    const cleaned = ctxLines.map((l) => l.replace(/^>\s?/, ""));

    const context: any = {
      notes: [],
      urls: [],
      tags: [],
      folders: [],
      webTabs: [],
    };

    let key: keyof typeof context | null = null;

    const parseNoteLink = (link: string) => {
      const match = link.match(/\[\[(.+?)\|(.+?)\]\]/);
      return match ? { path: match[1], basename: match[2] } : { path: link, basename: link };
    };

    for (const l of cleaned) {
      const top = l.match(/^- (.+)$/);
      if (top) {
        const section = top[1].toLowerCase();

        if (section === "notes") key = "notes";
        else if (section === "urls") key = "urls";
        else if (section === "tags") key = "tags";
        else if (section === "folders") key = "folders";
        else if (section === "web tabs") key = "webTabs";
        else key = null;

        continue;
      }

      const sub = l.match(/^\s*-\s+(.+)/);
      if (sub && key) {
        const val = sub[1].trim();

        if (key === "notes") {
          context.notes.push(parseNoteLink(val));
        } else if (key === "webTabs") {
          const urlMatch = val.match(/\]\((https?:\/\/[^)]+)\)/);
          if (urlMatch) {
            context.webTabs.push({ url: urlMatch[1] });
          }
        } else {
          context[key].push(val);
        }
      }
    }

    // Match old inline behavior: return undefined if empty
    if (
      context.notes.length === 0 &&
      context.urls.length === 0 &&
      context.tags.length === 0 &&
      context.folders.length === 0 &&
      context.webTabs.length === 0
    ) {
      return undefined;
    }

    return context;
  }

  /**
   * Parse context string back into a context object
   */
  private parseContextString(contextStr: string): any {
    const context: any = {
      notes: [],
      urls: [],
      tags: [],
      folders: [],
      webTabs: [],
    };

    // Split by | to get different context types
    const parts = contextStr.split(" | ");

    for (const part of parts) {
      const trimmed = part.trim();

      if (trimmed.startsWith("Notes: ")) {
        const notesStr = trimmed.substring(7); // Remove "Notes: "
        if (notesStr) {
          // Parse note paths and resolve to TFile objects
          context.notes = notesStr
            .split(", ")
            .map((pathStr) => {
              const trimmedPath = pathStr.trim();

              // Try to resolve by full path first (new format)
              const file = this.app.vault.getAbstractFileByPath(trimmedPath);
              if (file instanceof TFile) {
                return file;
              }

              // Backward compatibility: If path not found, try basename resolution
              const basename = trimmedPath.includes("/")
                ? trimmedPath.split("/").pop()!
                : trimmedPath;

              const matches = this.app.vault
                .getMarkdownFiles()
                .filter((f) => f.basename === basename);

              if (matches.length === 1) {
                logInfo(
                  `[ChatPersistenceManager] Resolved legacy basename "${basename}" to ${matches[0].path}`
                );
                return matches[0];
              } else if (matches.length > 1) {
                logWarn(
                  `[ChatPersistenceManager] Ambiguous basename "${basename}", skipping. Matches: ${matches.map((f) => f.path).join(", ")}`
                );
              } else {
                logWarn(`[ChatPersistenceManager] Note not found: ${trimmedPath}`);
              }

              return null;
            })
            .filter((note): note is TFile => note !== null);
        }
      } else if (trimmed.startsWith("URLs: ")) {
        const urlsStr = trimmed.substring(6); // Remove "URLs: "
        if (urlsStr) {
          context.urls = urlsStr.split(", ").map((url) => url.trim());
        }
      } else if (trimmed.startsWith("Web Tabs: ") || trimmed.startsWith("WebTabs: ")) {
        const webTabsStr = trimmed.startsWith("Web Tabs: ")
          ? trimmed.substring(10) // Remove "Web Tabs: "
          : trimmed.substring(9); // Remove "WebTabs: "
        if (webTabsStr) {
          context.webTabs = webTabsStr
            .split(", ")
            .map((url) => url.trim())
            .filter((url) => url.length > 0)
            .map((url) => ({ url }));
        }
      } else if (trimmed.startsWith("Tags: ")) {
        const tagsStr = trimmed.substring(6); // Remove "Tags: "
        if (tagsStr) {
          context.tags = tagsStr.split(", ").map((tag) => tag.trim());
        }
      } else if (trimmed.startsWith("Folders: ")) {
        const foldersStr = trimmed.substring(9); // Remove "Folders: "
        if (foldersStr) {
          context.folders = foldersStr.split(", ").map((folder) => folder.trim());
        }
      }
    }

    // Only return context if it has any content
    if (
      context.notes.length > 0 ||
      context.urls.length > 0 ||
      context.tags.length > 0 ||
      context.folders.length > 0 ||
      context.webTabs.length > 0
    ) {
      return context;
    }

    return undefined;
  }

  /**
   * Parse chat content in the legacy inline format using **user**: and **ai**: sender markers
   */
  private parseInlineChatFormat(content: string): ChatMessage[] {
    const messages: ChatMessage[] = [];
    const chatContent = this.stripFrontmatter(content);

    // Parse messages from the content
    // Look for the message pattern: **user**: or **ai**: followed by content
    const messagePattern = /\*\*(user|ai)\*\*: ([\s\S]*?)(?=(?:\n\*\*(?:user|ai)\*\*: )|$)/g;

    let match;
    while ((match = messagePattern.exec(chatContent)) !== null) {
      const sender = match[1] === "user" ? USER_SENDER : AI_SENDER;
      const fullContent = match[2].trim();

      // Split content into lines to extract timestamp, context, and message
      const contentLines = fullContent.split("\n");
      let messageText = fullContent;
      let timestamp = "Unknown time";
      let contextInfo: any = undefined;

      // Check for context and timestamp lines
      let endIndex = contentLines.length;

      // Check if last line is a timestamp
      if (contentLines[endIndex - 1]?.startsWith("[Timestamp: ")) {
        const timestampMatch = contentLines[endIndex - 1].match(/\[Timestamp: (.*?)\]/);
        if (timestampMatch) {
          timestamp = timestampMatch[1];
          endIndex--;
        }
      }

      // Check if second-to-last line is context
      if (endIndex > 0 && contentLines[endIndex - 1]?.startsWith("[Context: ")) {
        const contextMatch = contentLines[endIndex - 1].match(/\[Context: (.*?)\]/);
        if (contextMatch) {
          const contextStr = contextMatch[1];
          contextInfo = this.parseContextString(contextStr);
          endIndex--;
        }
      }

      // Message is everything before context and timestamp
      messageText = contentLines.slice(0, endIndex).join("\n").trim();

      // Strip old tool call markers and agent reasoning blocks from AI messages
      if (sender === AI_SENDER) {
        // Strip old tool call banners: <!--TOOL_CALL_START:...-->...<!--TOOL_CALL_END:...-->
        messageText = messageText.replace(
          /<!--TOOL_CALL_START:[^:]+:[^:]+:[^:]+:[^:]+:[^:]*:[^:]+-->[\s\S]*?<!--TOOL_CALL_END:[^:]+:[\s\S]*?-->/g,
          ""
        );
        // Strip agent reasoning blocks: <!--AGENT_REASONING:...-->
        const reasoningData = parseReasoningBlock(messageText);
        if (reasoningData) {
          messageText = reasoningData.contentAfter;
        }
        // Clean up any resulting multiple consecutive newlines
        messageText = messageText.replace(/\n{3,}/g, "\n\n").trim();
      }

      // Parse the timestamp
      let epoch: number | undefined;
      if (timestamp !== "Unknown time") {
        const date = new Date(timestamp);
        if (!isNaN(date.getTime())) {
          epoch = date.getTime();
        }
      }

      messages.push({
        message: messageText,
        sender,
        isVisible: true,
        timestamp: epoch
          ? {
              epoch,
              display: timestamp,
              fileName: "",
            }
          : null,
        context: contextInfo,
      });
    }

    return messages;
  }

  /**
   * Find a file by its epoch in the frontmatter
   */
  private async findFileByEpoch(epoch: number): Promise<TFile | null> {
    const files = await this.getChatHistoryFiles();

    for (const file of files) {
      // Try metadata cache first (works for non-hidden directories)
      let epochValue: unknown = this.app.metadataCache.getFileCache(file)?.frontmatter?.epoch;

      // Fallback for hidden directory files: read frontmatter via adapter
      if (epochValue === undefined) {
        try {
          const adapterFm = await readFrontmatterViaAdapter(this.app, file.path);
          if (adapterFm?.epoch) epochValue = Number(adapterFm.epoch);
        } catch {
          continue;
        }
      }

      const frontmatterEpoch =
        typeof epochValue === "number"
          ? epochValue
          : typeof epochValue === "string"
            ? Number(epochValue)
            : undefined;
      if (
        typeof frontmatterEpoch === "number" &&
        !Number.isNaN(frontmatterEpoch) &&
        frontmatterEpoch === epoch
      ) {
        return file;
      }
    }

    return null;
  }

  /**
   * Generate AI topic for the conversation
   */
  private async generateAITopic(messages: ChatMessage[]): Promise<string | undefined> {
    if (!this.chainManager) {
      return undefined;
    }

    try {
      const chatModel = this.chainManager.chatModelManager.getChatModel();
      if (!chatModel) {
        return undefined;
      }

      // Constants for topic generation
      const TOPIC_GENERATION_MESSAGE_LIMIT = 6;
      const TOPIC_GENERATION_CHAR_LIMIT = 200;

      // Get conversation content for topic generation - using reduce for efficiency
      const conversationSummary = messages.reduce((acc, m, i) => {
        if (i >= TOPIC_GENERATION_MESSAGE_LIMIT) return acc;
        return (
          acc +
          (acc ? "\n" : "") +
          `${m.sender}: ${m.message.slice(0, TOPIC_GENERATION_CHAR_LIMIT)}`
        );
      }, "");

      const prompt = `Generate a concise title (max 5 words) for this conversation based on its content. Return only the title without any explanation or quotes.

Conversation:
${conversationSummary}`;

      const response = await chatModel.invoke(prompt);
      const responseContent =
        typeof response === "string"
          ? response
          : ((response as { content?: unknown; text?: unknown }).content ??
            (response as { content?: unknown; text?: unknown }).text ??
            response);
      const topic = extractTextFromChunk(responseContent)
        .trim()
        .replace(/^["']|["']$/g, "") // Remove quotes if present
        .replace(/[\\/:*?"<>|]/g, "") // Remove invalid filename characters
        .slice(0, 50); // Limit length

      return topic || undefined;
    } catch (error) {
      logError("[ChatPersistenceManager] Error generating AI topic:", error);
      return undefined;
    }
  }

  /**
   * Generate a file name for the chat.
   * @param messages - The conversation messages used to derive the topic.
   * @param firstMessageEpoch - Epoch timestamp of the first message in the chat.
   * @param topic - Optional pre-computed topic to use for the filename.
   */
  private generateFileName(
    messages: ChatMessage[],
    firstMessageEpoch: number,
    topic?: string
  ): string {
    const settings = getSettings();
    const formattedDateTime = formatDateTime(new Date(firstMessageEpoch));
    const timestampFileName = formattedDateTime.fileName;

    // Use provided topic or fall back to first 10 words
    let topicForFilename: string;
    if (topic) {
      topicForFilename = topic;
    } else {
      // Get the first user message
      const firstUserMessage = messages.find((message) => message.sender === USER_SENDER);

      // Get the first 10 words from the first user message and sanitize them
      topicForFilename = firstUserMessage
        ? firstUserMessage.message
            // Remove Obsidian wiki link brackets while preserving inner text: [[Title]] -> Title
            .replace(/\[\[([^\]]+)\]\]/g, "$1")
            // Remove any remaining square brackets or braces
            .replace(/[{}[\]]/g, "")
            // Now split to first 10 words
            .split(/\s+/)
            .slice(0, 10)
            .join(" ")
            // Remove invalid filename characters (including control chars)
            // eslint-disable-next-line no-control-regex
            .replace(/[\\/:*?"<>|\x00-\x1F]/g, "")
            .trim() || "Untitled Chat"
        : "Untitled Chat";
    }

    // Parse the custom format and replace variables
    let customFileName = settings.defaultConversationNoteName || "{$date}_{$time}__{$topic}";

    // Get the current project prefix if any
    const currentProject = getCurrentProject();
    const filePrefix = currentProject ? `${currentProject.id}__` : "";

    // Calculate fixed components in bytes
    const extensionBytes = getUtf8ByteLength(".md");
    const filePrefixBytes = getUtf8ByteLength(filePrefix);

    // Calculate the custom format overhead (everything except {$topic})
    const formatOverhead = customFileName
      .replace("{$topic}", "")
      .replace("{$date}", timestampFileName.split("_")[0])
      .replace("{$time}", timestampFileName.split("_")[1]);
    const formatOverheadBytes = getUtf8ByteLength(formatOverhead);

    // Calculate the maximum bytes available for the topic
    const topicByteBudget = Math.max(
      20, // Minimum 20 bytes for topic to ensure at least some meaningful text
      SAFE_FILENAME_BYTE_LIMIT - extensionBytes - filePrefixBytes - formatOverheadBytes
    );

    // Replace spaces with underscores and truncate to byte limit
    const topicWithUnderscores = topicForFilename.replace(/\s+/g, "_");
    const truncatedTopic = truncateToByteLimit(topicWithUnderscores, topicByteBudget);

    // Create the file name with the truncated topic
    customFileName = customFileName
      .replace("{$topic}", truncatedTopic)
      .replace("{$date}", timestampFileName.split("_")[0])
      .replace("{$time}", timestampFileName.split("_")[1]);

    // Sanitize the final filename (replace any illegal chars with underscore)
    // Also remove leftover square brackets which are illegal on some platforms
    // eslint-disable-next-line no-control-regex
    const sanitizedFileName = customFileName
      .replace(/\[\[([^\]]+)\]\]/g, "$1")
      .replace(/[{}[\]]/g, "_")
      // eslint-disable-next-line no-control-regex
      .replace(/[\\/:*?"<>|\x00-\x1F]/g, "_");

    // Final safety check: ensure the complete basename fits within the limit
    const baseNameWithPrefix = `${filePrefix}${sanitizedFileName}.md`;
    if (getUtf8ByteLength(baseNameWithPrefix) > SAFE_FILENAME_BYTE_LIMIT) {
      // If still too long, truncate the entire filename more aggressively
      const availableForBasename = SAFE_FILENAME_BYTE_LIMIT - extensionBytes - filePrefixBytes;
      const truncatedBasename = truncateToByteLimit(sanitizedFileName, availableForBasename);
      return `${settings.defaultSaveFolder}/${filePrefix}${truncatedBasename}.md`;
    }

    return `${settings.defaultSaveFolder}/${baseNameWithPrefix}`;
  }

  /**
   * Generate the full note content with frontmatter
   */
  private generateNoteContent(
    chatContent: string,
    firstMessageEpoch: number,
    modelKey: string,
    topic?: string,
    lastAccessedAt?: number
  ): string {
    const settings = getSettings();
    const currentProject = getCurrentProject();

    return `---
epoch: ${firstMessageEpoch}
modelKey: "${escapeYamlString(modelKey)}"${topic ? `\ntopic: "${topic}"` : ""}${
      lastAccessedAt ? `\nlastAccessedAt: ${lastAccessedAt}` : ""
    }${currentProject ? `\nprojectId: ${currentProject.id}` : ""}${
      currentProject ? `\nprojectName: ${currentProject.name}` : ""
    }
tags:
  - ${settings.defaultConversationTag}
---

${chatContent}`;
  }

  /**
   * Trigger asynchronous topic generation and apply it to the saved note once available
   */
  private generateTopicAsyncIfNeeded(
    messages: ChatMessage[],
    file: TFile | null,
    existingTopic?: string
  ): void {
    const settings = getSettings();

    if (!settings.generateAIChatTitleOnSave || !file || existingTopic) {
      return;
    }

    void (async () => {
      try {
        const topic = await this.generateAITopic(messages);
        if (!topic) {
          return;
        }
        await this.applyTopicToFrontmatter(file, topic);
        await this.renameFileToMatchTopic(file, topic);
      } catch (error) {
        logError("[ChatPersistenceManager] Error during async topic generation:", error);
      }
    })();
  }

  /**
   * Apply the AI-generated topic to the note's YAML frontmatter
   */
  private async applyTopicToFrontmatter(file: TFile, topic: string): Promise<void> {
    try {
      await patchFrontmatter(this.app, file.path, { topic: topic.trim() });
      logInfo(`[ChatPersistenceManager] Applied AI topic to chat file: ${file.path}`);
    } catch (error) {
      logError("[ChatPersistenceManager] Error applying AI topic to file:", error);
    }
  }

  /**
   * Determine whether an error corresponds to an ENAMETOOLONG filesystem failure.
   * @param error - The thrown error.
   * @returns True when the error message indicates a name-length constraint violation.
   */
  private isNameTooLongError(error: unknown): boolean {
    if (!error) {
      return false;
    }

    const message = error instanceof Error ? error.message : String(error);
    const normalized = message.toLowerCase();
    return normalized.includes("enametoolong") || normalized.includes("name too long");
  }

  /**
   * Determine if an error indicates an Obsidian file-exists conflict.
   */
  private isFileAlreadyExistsError(error: unknown): boolean {
    if (!error) {
      return false;
    }
    const message = error instanceof Error ? error.message : String(error);
    return message.toLowerCase().includes("already exists");
  }

  /**
   * Strip YAML frontmatter from the chat content
   */
  private stripFrontmatter(content: string): string {
    const match = content.match(/^---\n([\s\S]*?)\n---/);
    return match ? content.slice(match[0].length).trim() : content;
  }
}
