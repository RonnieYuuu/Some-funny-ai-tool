import queue
import tempfile
import time
import os
import numpy as np
import pygame
import requests
import sounddevice as sd
import soundfile as sf
import whisper
from gtts import gTTS
import tkinter as tk
from tkinter import scrolledtext, ttk
import threading

# 配置参数
SAMPLE_RATE = 16000
CHUNK_DURATION = 3
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# 支持的语言
LANGUAGES = {
    "English": {
        "code": "en",
        "system_prompt": "You are a friendly English tutor. Respond concisely in under 100 words in English."
    },
    "Deutsch": {
        "code": "de",
        "system_prompt": "Du bist ein freundlicher Deutschlehrer. Antworte kurz und prägnant in maximal 100 Wörtern auf Deutsch."
    },
    "中文": {
        "code": "zh",
        "system_prompt": "你是一位友好的中文老师。请用100字以内的中文简洁回答。"
    }
}

# 语音检测参数
SILENCE_THRESHOLD = 0.015
MIN_SPEECH_DURATION = 0.3
MIN_SILENCE_DURATION = 0.8
PRE_SPEECH_BUFFER = 0.5

# 在配置参数部分添加可用模型列表
AVAILABLE_MODELS = {
    "Llama 3.1": "llama3.1",
    "DeepSeek-R1": "deepseek-R1"
}


class OllamaClient:
    def __init__(self, model_name="llama3.1"):
        self.model_name = model_name
        self.api_url = OLLAMA_API_URL

    def get_response(self, messages):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                response_data = response.json()
                return response_data['message']['content']
            else:
                print(f"Ollama API Error {response.status_code}: {response.text}")
                return None

        except Exception as e:
            print(f"Ollama API Request Failed: {str(e)}")
            return None


class VoiceChatbot:
    def __init__(self, root=None, model_name="llama3.1"):  # 默认改为 llama3.1
        # 初始化语音模型
        self.whisper_model = whisper.load_model("tiny")  # 使用更小的模型提高速度

        # 初始化Ollama客户端
        self.llm_client = OllamaClient(model_name)

        # 音频队列系统
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()

        # 初始化TTS引擎
        pygame.mixer.init()

        # 当前语言设置
        self.current_language = "English"

        # 对话上下文管理
        self.context = [
            {"role": "system", "content": LANGUAGES[self.current_language]["system_prompt"]}
        ]

        # 状态标志
        self.is_recording = False
        self.is_playing = False
        self.recording_stream = None
        self.speech_detected = False
        self.ready_for_new_input = True

        # 临时文件管理
        self.temp_dir = tempfile.mkdtemp()

        # UI组件
        self.root = root
        if self.root:
            self.setup_ui()

        # 音量监测变量
        self.current_volume = 0

        # 添加新的控制标志
        self.processing_enabled = True

    def setup_ui(self):
        """设置GUI界面"""
        self.root.title("Voice Chat Assistant")
        self.root.geometry("600x500")

        # 添加模型选择
        self.model_frame = ttk.Frame(self.root)
        self.model_frame.pack(padx=10, pady=5, fill=tk.X)

        ttk.Label(self.model_frame, text="Model:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value="Llama 3.1")
        self.model_combo = ttk.Combobox(self.model_frame,
                                        textvariable=self.model_var,
                                        values=list(AVAILABLE_MODELS.keys()),
                                        state="readonly")
        self.model_combo.pack(side=tk.LEFT, padx=5)
        self.model_combo.bind('<<ComboboxSelected>>', self.change_model)

        # 聊天记录区域
        self.chat_frame = ttk.Frame(self.root)
        self.chat_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.chat_display = scrolledtext.ScrolledText(self.chat_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.chat_display.pack(fill=tk.BOTH, expand=True)

        # 控制区域
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(padx=10, pady=10, fill=tk.X)

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT, padx=5)

        self.record_button = ttk.Button(self.control_frame, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT, padx=5)

        # 音量指示器
        self.volume_frame = ttk.Frame(self.root)
        self.volume_frame.pack(padx=10, pady=5, fill=tk.X)

        self.volume_meter = ttk.Progressbar(self.volume_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.volume_meter.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # 添加语言选择
        self.lang_frame = ttk.Frame(self.root)
        self.lang_frame.pack(padx=10, pady=5, fill=tk.X)

        ttk.Label(self.lang_frame, text="Language:").pack(side=tk.LEFT, padx=5)
        self.language_var = tk.StringVar(value=self.current_language)
        self.language_combo = ttk.Combobox(self.lang_frame,
                                           textvariable=self.language_var,
                                           values=list(LANGUAGES.keys()),
                                           state="readonly")
        self.language_combo.pack(side=tk.LEFT, padx=5)
        self.language_combo.bind('<<ComboboxSelected>>', self.change_language)

        # 启动音量监测
        self.update_volume_meter()

    def update_volume_meter(self):
        """更新音量指示器"""
        if self.is_recording:
            volume_percentage = min(100, int(self.current_volume * 100))
            self.volume_meter["value"] = volume_percentage
        else:
            self.volume_meter["value"] = 0

        self.root.after(100, self.update_volume_meter)

    def toggle_recording(self):
        """切换录音状态"""
        if self.is_recording:
            self.stop_recording()
            self.record_button.config(text="Start Recording")
            self.status_var.set("Stopped recording")
        else:
            self.start_recording()
            self.record_button.config(text="Stop Recording")
            self.status_var.set("正在认真听你说呢...")

    def _recording_callback(self, indata, frames, time, status):
        """音频输入回调函数"""
        if status:
            print(f"Recording error: {status}")

        volume = np.max(np.abs(indata))
        self.current_volume = volume * 2

        # 只有在允许处理时才添加到队列
        if self.processing_enabled and not self.is_playing:
            self.audio_queue.put((indata.copy(), volume))

    def start_recording(self):
        """启动录音"""
        if self.is_recording:
            return

        print("\n🎤 Recording started...")
        self.is_recording = True
        self.ready_for_new_input = True

        self.recording_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=self._recording_callback,
            blocksize=int(SAMPLE_RATE * 0.1)
        )
        self.recording_stream.start()

        # 启动处理线程
        self.processing_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.processing_thread.start()

    def stop_recording(self):
        """停止录音"""
        if not self.is_recording:
            return

        self.is_recording = False
        if self.recording_stream:
            self.recording_stream.stop()
            self.recording_stream.close()
            self.recording_stream = None

    def process_audio(self):
        """处理音频队列"""
        accumulated_audio = []
        accumulated_time = 0
        speech_started = False
        speech_duration = 0
        silence_duration = 0

        while self.is_recording:
            try:
                # 只在允许处理时获取音频
                if not self.processing_enabled:
                    time.sleep(0.1)
                    continue

                audio_chunk, volume = self.audio_queue.get(timeout=0.5)
                chunk_duration = len(audio_chunk) / SAMPLE_RATE
                accumulated_time += chunk_duration

                if volume > SILENCE_THRESHOLD:
                    if not speech_started and accumulated_time >= PRE_SPEECH_BUFFER:
                        speech_started = True
                        self.speech_detected = True

                    if speech_started:
                        speech_duration += chunk_duration
                        silence_duration = 0
                else:
                    if speech_started:
                        silence_duration += chunk_duration

                accumulated_audio.append(audio_chunk)

                should_process = (speech_started and
                                  speech_duration >= MIN_SPEECH_DURATION and
                                  silence_duration >= MIN_SILENCE_DURATION)

                if should_process:
                    self.process_accumulated_audio(accumulated_audio)
                    accumulated_audio = []
                    accumulated_time = 0
                    speech_started = False
                    speech_duration = 0
                    silence_duration = 0
                    self.speech_detected = False

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing Error: {str(e)}")
                time.sleep(1)

    def process_accumulated_audio(self, audio_chunks):
        """处理累积的音频数据"""
        if not audio_chunks or not self.processing_enabled:
            return

        try:
            # 暂时禁用处理以防止录入AI回答
            self.processing_enabled = False

            # 更新状态为转录中
            if self.root:
                self.status_var.set("你的声音本AI听得一清二楚...")

            # 使用更快的音频处理
            audio_data = np.concatenate(audio_chunks)
            temp_file = os.path.join(self.temp_dir, "temp_audio.wav")
            sf.write(temp_file, audio_data, SAMPLE_RATE)

            # 清空音频队列，防止处理到AI的回答
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

            # 添加超时控制
            result = self.whisper_model.transcribe(
                temp_file,
                language=LANGUAGES[self.current_language]["code"],
                temperature=0.0,
                no_speech_threshold=0.3
            )
            user_text = result["text"].strip()

            if user_text:
                print(f"\n👤 User: {user_text}")
                if self.root:
                    self.update_chat_display("user", user_text)

                self.context.append({"role": "user", "content": user_text})

                # 获取AI响应
                ai_response = self.llm_client.get_response(self.context)

                if ai_response:
                    print(f"\n🤖 AI: {ai_response}")
                    if self.root:
                        self.update_chat_display("assistant", ai_response)
                        self.status_var.set("帅气回答中...")

                    self.context.append({"role": "assistant", "content": ai_response})

                    # 文本转语音并播放
                    self.text_to_speech(ai_response)

        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            if self.root:
                self.status_var.set(f"Error: {str(e)}")
        finally:
            # 重新启用处理
            self.processing_enabled = True
            # 恢复录音状态显示
            if self.root and self.is_recording:
                self.status_var.set("正在认真听你说呢...")

    def text_to_speech(self, text):
        """文本转语音"""
        try:
            # 确保处理被禁用
            self.processing_enabled = False

            if self.root:
                self.status_var.set("夜以继日地加工语料...")

            # 清空音频队列
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

            # 使用gTTS生成语音
            response_file = os.path.join(self.temp_dir, "response.mp3")
            tts = gTTS(text=text, lang=LANGUAGES[self.current_language]["code"], slow=False)
            tts.save(response_file)

            self.is_playing = True

            if self.root:
                self.status_var.set("AI正在帅气回答...")

            # 播放音频
            pygame.mixer.music.load(response_file)
            pygame.mixer.music.play()

            # 等待播放完成
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            time.sleep(0.5)  # 添加短暂延迟
            self.is_playing = False

            # 再次清空音频队列
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

        except Exception as e:
            print(f"TTS Error: {str(e)}")
            if self.root:
                self.status_var.set(f"TTS Error: {str(e)}")
        finally:
            self.is_playing = False
            # 重新启用处理
            self.processing_enabled = True
            # 恢复录音状态显示
            if self.root and self.is_recording:
                self.status_var.set("正在认真听你说呢...")

    def update_chat_display(self, speaker, text):
        """更新聊天显示"""
        if not self.root:
            return

        self.chat_display.config(state=tk.NORMAL)

        if speaker == "user":
            self.chat_display.insert(tk.END, f"\n👤 You: {text}\n", "user")
        else:
            self.chat_display.insert(tk.END, f"\n🤖 AI: {text}\n", "assistant")

        self.chat_display.tag_configure("user", foreground="#0000FF")
        self.chat_display.tag_configure("assistant", foreground="#FF0000")

        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def change_language(self, event=None):
        """切换语言"""
        new_language = self.language_var.get()
        if new_language != self.current_language:
            self.current_language = new_language
            # 更新系统提示
            self.context = [
                {"role": "system", "content": LANGUAGES[self.current_language]["system_prompt"]}
            ]
            if self.root:
                self.status_var.set(f"Language changed to {new_language}")
            print(f"Language changed to {new_language}")

    def change_model(self, event=None):
        """切换模型"""
        selected_model = self.model_var.get()
        model_name = AVAILABLE_MODELS[selected_model]
        self.llm_client = OllamaClient(model_name)

        if self.root:
            self.status_var.set(f"Model changed to {selected_model}")
        print(f"Model changed to {selected_model}")

        # 重置对话上下文
        self.context = [
            {"role": "system", "content": LANGUAGES[self.current_language]["system_prompt"]}
        ]

    def run(self):
        """运行主程序"""
        if self.root:
            self.root.mainloop()
        else:
            try:
                self.start_recording()
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                self.stop_recording()
                pygame.mixer.quit()


if __name__ == "__main__":
    try:
        root = tk.Tk()
        print("Starting Voice Chat Assistant (GUI Mode)...")
        chatbot = VoiceChatbot(root, model_name="llama3.1")  # 设置默认模型
        chatbot.run()
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        print("Starting Voice Chat Assistant (Console Mode)...")
        chatbot = VoiceChatbot(model_name="llama3.1")  # 设置默认模型
        chatbot.run()
