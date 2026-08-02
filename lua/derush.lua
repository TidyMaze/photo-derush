--[[
  Photo-Derush Darktable Lua Plugin
  Integrates photo-derush ML active learning & burst grouping into Lighttable
--]]

local dt = require "darktable"
local dt_sys = require "darktable.sys"

-- Plugin registration
dt.register_event("shortcut", function(event, shortcut)
    dt.print("Photo-Derush shortcut triggered")
end, "Photo-Derush Action")

local function run_derush_command(cmd_name, folder_path)
    local python_bin = "py -m poetry run python"
    local script_path = dt_sys.get_user_config_dir() .. "/lua/derush/cli_bridge.py"
    local command = string.format("%s %s %s --directory %q", python_bin, script_path, cmd_name, folder_path)

    local handle = io.popen(command)
    if not handle then return nil end
    local result = handle:read("*a")
    handle:close()
    return result
end

-- UI Panel Widget in Lighttable
local widget_box = dt.new_widget("box") {
    orientation = "vertical",
    dt.new_widget("label") { label = "Photo-Derush ML Assistant" },
    dt.new_widget("button") {
        label = "Run Burst Grouping (pHash)",
        tooltip = "Group duplicate and burst photos automatically",
        connect_callback = function()
            local images = dt.gui.action("selection/get_selected", "")
            dt.print("Running Derush burst grouping...")
        end
    }
}

dt.register_lib("derush_panel", "Photo-Derush", true, false, {
    [dt.gui.views.lighttable] = {"selection", 100}
}, widget_box)
