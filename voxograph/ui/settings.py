"""voxograph.ui.settings - MainWindow SettingsMixin methods."""
from voxograph.core import (
    save_settings,
)
from voxograph.core.constants import (
    CHATTERBOX_TURBO_DEFAULT_TEMPERATURE, CHATTERBOX_TURBO_DEFAULT_TOP_P,
    CHATTERBOX_TURBO_DEFAULT_TOP_K, CHATTERBOX_TURBO_DEFAULT_REPETITION_PENALTY,
)


class SettingsMixin:
    def save_current_settings(self):
        """Save current GUI settings via core's save_settings."""
        output_folder = self.output_folder_text_ctrl.GetValue() if (hasattr(self, 'output_folder_text_ctrl') and self.output_folder_text_ctrl) else self.settings.get('output_folder', '.')
        voice = self.get_selected_voice() if hasattr(self, 'selected_voice') else self.settings.get('voice', 'af_heart')
        speed = self.selected_speed if hasattr(self, 'selected_speed') else self.settings.get('speed', 1.0)
        gemini_api_key = self.ai_api_key_text_ctrl.GetValue() if (hasattr(self, 'ai_api_key_text_ctrl') and self.ai_api_key_text_ctrl) else self.settings.get('gemini_api_key', '')
        gemini_api_key = ' '.join(gemini_api_key.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ').split())
        gemini_model = self.ai_model_dropdown.GetValue() if (hasattr(self, 'ai_model_dropdown') and self.ai_model_dropdown) else self.settings.get('gemini_model', 'gemini-3.1-flash-lite')
        gemini_enabled = self.ai_enabled_checkbox.GetValue() if (hasattr(self, 'ai_enabled_checkbox') and self.ai_enabled_checkbox) else self.settings.get('gemini_enabled', False)
        last_open_dir = self.last_open_dir if hasattr(self, 'last_open_dir') else self.settings.get('last_open_dir', '')
        tts_engine = self.get_selected_tts_engine() if hasattr(self, 'chatterbox_radio') else self.settings.get('tts_engine', 'kokoro')
        chatterbox_ref_audio = self.selected_ref_audio if hasattr(self, 'selected_ref_audio') else self.settings.get('chatterbox_ref_audio', '')
        chatterbox_device = self.get_chatterbox_device() if hasattr(self, 'cuda_radio') else self.settings.get('chatterbox_device', 'cuda')
        chatterbox_voice_source = self.get_preset_source() if hasattr(self, 'preset_radio') else self.settings.get('chatterbox_voice_source', 'preset')
        chatterbox_exaggeration = self.get_chatterbox_exaggeration() if hasattr(self, 'exaggeration_spin') else self.settings.get('chatterbox_exaggeration', 0.5)
        chatterbox_cfg_weight = self.get_chatterbox_cfg_weight() if hasattr(self, 'cfg_weight_spin') else self.settings.get('chatterbox_cfg_weight', 0.5)
        chatterbox_model = self.get_chatterbox_model() if hasattr(self, 'model_dropdown') else self.settings.get('chatterbox_model', 'multilingual')
        chatterbox_turbo_temperature = self.get_chatterbox_turbo_temperature() if hasattr(self, 'turbo_temperature_spin') else self.settings.get('chatterbox_turbo_temperature', CHATTERBOX_TURBO_DEFAULT_TEMPERATURE)
        chatterbox_turbo_top_p = self.get_chatterbox_turbo_top_p() if hasattr(self, 'turbo_top_p_spin') else self.settings.get('chatterbox_turbo_top_p', CHATTERBOX_TURBO_DEFAULT_TOP_P)
        chatterbox_turbo_top_k = self.get_chatterbox_turbo_top_k() if hasattr(self, 'turbo_top_k_spin') else self.settings.get('chatterbox_turbo_top_k', CHATTERBOX_TURBO_DEFAULT_TOP_K)
        chatterbox_turbo_repetition_penalty = self.get_chatterbox_turbo_repetition_penalty() if hasattr(self, 'turbo_repetition_penalty_spin') else self.settings.get('chatterbox_turbo_repetition_penalty', CHATTERBOX_TURBO_DEFAULT_REPETITION_PENALTY)
        voice_samples_dir = self.settings.get('voice_samples_dir', '')

        save_settings(
            output_folder=output_folder,
            voice=voice,
            speed=speed,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            gemini_enabled=gemini_enabled,
            last_open_dir=last_open_dir,
            tts_engine=tts_engine,
            chatterbox_ref_audio=chatterbox_ref_audio,
            chatterbox_device=chatterbox_device,
            chatterbox_voice_source=chatterbox_voice_source,
            voice_samples_dir=voice_samples_dir,
            chatterbox_exaggeration=chatterbox_exaggeration,
            chatterbox_cfg_weight=chatterbox_cfg_weight,
            chatterbox_model=chatterbox_model,
            chatterbox_turbo_temperature=chatterbox_turbo_temperature,
            chatterbox_turbo_top_p=chatterbox_turbo_top_p,
            chatterbox_turbo_top_k=chatterbox_turbo_top_k,
            chatterbox_turbo_repetition_penalty=chatterbox_turbo_repetition_penalty,
        )
    def get_selected_voice(self):
        """Return just the voice code, stripping any leading flag emoji."""
        parts = self.selected_voice.split(' ')
        return parts[1] if len(parts) > 1 else self.selected_voice
    def get_selected_speed(self):
        return float(self.selected_speed)
