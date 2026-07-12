import inspect

from ui.connection_controller import ConnectionController
from ui.capture_controller import CaptureController
from ui.main_window_actions import MainWindowActions
from ui.motion_controller import MotionController
from ui.plot_renderer import PlotRenderer


CONNECTION_METHODS = ['_on_connect_clicked', '_event_handler', '_watchdog_loop', '_start_watchdog', '_stop_watchdog', '_mark_connection_lost', '_on_connection_lost_ui', '_attempt_connection_with_timeout', '_cleanup_connection_async', 'do_connect', '_on_connect_progress', '_on_connect_result', 'do_disconnect']
CAPTURE_METHODS = ['_on_source_changed', '_get_drive_sample_time_units', '_update_drive_info_label', '_on_drive_trigger_changed', 'start_capture', '_start_drive_scope_capture', '_drive_scope_capture_thread', '_arm_and_wait_for_external_trigger', '_scope_single_external_trigger_thread', '_scope_continuous_external_trigger_thread', '_scope_single_shot_thread', '_scope_continuous_thread', '_push_data', '_push_segment_break', '_on_capture_progress', '_on_capture_status', '_on_capture_stopped', '_on_update_timer', 'stop_capture']
PLOT_METHODS = ['_create_scope_plot', '_on_plot_double_click', '_recreate_subplots', '_configure_plot', '_configure_fft_plot', '_on_manual_range_change', '_reposition_stats_texts', '_flush_stats_reposition', '_update_curve_detail', '_flush_curve_detail', '_do_update_curve_detail', '_on_xy_manual_zoom', '_add_hover_elements_to_plot', '_on_main_plot_mouse_moved', '_flush_hover_update', '_open_trace_window', '_open_compare', '_on_compare_closed', '_push_compare_data', '_compute_fft_payload_for_trace', '_toggle_cursors', '_init_cursor_positions', '_add_cursors_to_plots', '_remove_cursors_from_plots', '_on_cursor_line_moved', '_get_nearest_index', '_update_cursor_readout', '_setup_3d_view', '_build_3d_colorbar', '_update_colorbar_range', '_update_x_links', '_on_lock_x_changed', '_on_plot_mode_changed', '_on_path_view_scale_changed', '_sync_path_view_scale', '_update_path_info_label', 'add_trace', 'on_trace_changed', '_on_pin_toggled', 'get_enabled_traces', '_render_plots', 'toggle_auto_scroll', '_update_auto_scroll_button', '_fit_all_data', 'clear_data', '_apply_plot_settings']
MOTION_METHODS = ['_open_motion_window', '_on_motion_enable_requested', '_on_motion_start_requested', '_on_motion_stop_requested', '_sync_motion_connection_state', '_reset_motion_on_disconnect', '_disable_motion_axes_before_disconnect']
ACTION_METHODS = ['take_screenshot', 'export_to_csv', 'export_html_report', '_show_html_report_dialog', '_report_trace_context', '_report_controller_metadata', '_report_drive_metadata', '_report_drive_profiles', 'import_from_csv', '_get_profile_names', '_save_profile', '_load_profile', '_delete_profile', '_rename_profile', '_rebuild_profiles_menu', '_show_save_profile_dialog', '_show_manage_profiles_dialog', '_create_menu_bar', '_show_help', '_show_about', '_toggle_tuner_panel', '_toggle_measurement_panel', '_sync_measurement_panel', '_open_ethercat_map', '_get_scope_data_for_ai', 'open_settings', '_load_settings', '_save_settings']


def bind_main_window_controllers(window):
    window.connection_controller = ConnectionController(window)
    window.capture_controller = CaptureController(window)
    window.plot_renderer = PlotRenderer(window)
    window.motion_controller = MotionController(window)
    window.actions_controller = MainWindowActions(window)

    def bind_proxy(method_name, controller):
        method = getattr(controller, method_name)
        signature = inspect.signature(method)
        accepts_varargs = any(
            p.kind == inspect.Parameter.VAR_POSITIONAL
            for p in signature.parameters.values()
        )
        positional_count = sum(
            p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            for p in signature.parameters.values()
        )

        def proxy(*args, **kwargs):
            if accepts_varargs:
                return method(*args, **kwargs)
            return method(*args[:positional_count], **kwargs)

        proxy.__name__ = method_name
        proxy.__qualname__ = f"{type(window).__name__}.{method_name}"
        setattr(window, method_name, proxy)

    for method_name in CONNECTION_METHODS:
        bind_proxy(method_name, window.connection_controller)
    for method_name in CAPTURE_METHODS:
        bind_proxy(method_name, window.capture_controller)
    for method_name in PLOT_METHODS:
        bind_proxy(method_name, window.plot_renderer)
    for method_name in MOTION_METHODS:
        bind_proxy(method_name, window.motion_controller)
    for method_name in ACTION_METHODS:
        bind_proxy(method_name, window.actions_controller)
