DB_NAME = 'bupt'
ALG_TYPES = ('svm', 'dt', 'gauss', 'log')

NGRAM = 3
NGRAM_THRE = 100
TNGRAM_THRE = 2000
NGRAM_MOST_COMMON = 120
DB_REGEX = '((SELECT|select)\s[\w\*\)\(\,\s]+\s(FROM|from)?\s[\w]+)| ' \
           '((UPDATE|update)\s[\w]+\s(SET|set)\s[\w\,\'\=]+)| ' \
           '((INSERT|insert)\s(INTO|into)\s[\d\w]+[\s\w\d\)\(\,]*\s(VALUES|values)\s\([\d\w\'\,\)]+)| ' \
           '((DELETE|delete)\s(FROM|from)\s[\d\w\'\=]+)'

SPECIAL_STRINGS = ('com.metasploit.stage.PayloadTrustManager', )

PERMISSIONS = ('android.permission.BIND_WALLPAPER',
               'com.google.android.providers.gsf.permission.READ_GSERVICES',
               'android.permission.FORCE_BACK',
               'android.permission.READ_CALENDAR',
               'android.permission.READ_FRAME_BUFFER',
               'org.gnucash.android.permission.RECORD_TRANSACTION',
               'android.permission.DEVICE_POWER',
               'com.motorola.dlauncher.permission.INSTALL_SHORTCUT',
               'android.permission.READ_SYNC_STATS',
               'android.permission.ACCESS_COARSE_UPDATES',
               'thinkpanda.permission.CLEAR_MISSED_CALL',
               'android.permission.INTERNET',
               'android.permission.CHANGE_CONFIGURATION',
               'android.permission.CLEAR_APP_USER_DATA',
               'com.estrongs.android.pop.PERMISSION',
               'com.nilhcem.frcndict.permission.C2D_MESSAGE',
               'com.lge.launcher.permission.INSTALL_SHORTCUT',
               'android.permission.HARDWARE_TEST',
               'org.linphone.permission.C2D_MESSAGE',
               'com.android.browser.permission.WRITE_HISTORY_BOOKMARKS',
               'android.permission.ADD_SYSTEM_SERVICE',
               'com.android.launcher.permission.INSTALL_SHORTCUT',
               'org.openintents.ssh.permission.ACCESS_SSH_AGENT',
               'android.permission.WRITE_CALL_LOG',
               'android.permission.CHANGE_WIFI_MULTICAST_STATE',
               'android.permission.ACCESS_GPS',
               'info.guardianproject.otr.app.providers.imps.permission.READ_ONLY',
               'org.kontalk.permission.C2D_MESSAGE',
               'org.projectmaxs.permission.USE_TRANSPORT',
               'android.permission.BIND_INPUT_METHOD',
               'android.permission.CHANGE_WIMAX_STATE',
               'org.mariotaku.twidere.WRITE_DATABASES',
               'android.permission.WRITE_SYNC_SETTINGS',
               'com.samsungmobileusa.magnacarta.permission.C2D_MESSAGE',
               'android.permission.WRITE_USER_DICTIONARY',
               'com.sonyericsson.extras.liveview.permission.LIVEVIEW_API',
               'org.mozilla.firefox.permissions.FORMHISTORY_PROVIDER',
               'org.fdroid.k9.permission.REMOTE_CONTROL',
               'android.permission.WRITE_GSERVICES',
               'com.software.application.permission.C2D_MESSAGE',
               'android.permission.INJECT_EVENTS',
               'com.fsck.k9.permission.READ_MESSAGES',
               'org.koxx.k9ForPureWidget.permission.DELETE_MESSAGES',
               'org.mozilla.firefox_sync.permission.PER_ACCOUNT_TYPE',
               'android.permission.STORAGE',
               'ru.gelin.android.weather.notification.START_UPDATE_SERVICE',
               'android.permission.WRITE_SECURE_SETTINGS',
               'org.projectmaxs.permission.WRITE_SMS',
               'android.permission.CALL_PRIVILEGED',
               'android.permission.READ_OWNER_DATA',
               'com.dririan.RingyDingyDingy.HANDLE_INTERNAL_COMMAND',
               'android.permission.SYSTEM_ALERT_WINDOW',
               'android.permission.ACCESS_LOCATION_EXTRA_COMMANDS',
               'de.shandschuh.sparserss.READFEEDS',
               'android.permission.DUMP',
               'org.eehouse.android.xw4.permission.C2D_MESSAGE',
               'com.google.android.providers.gmail.permission.READ_GMAIL',
               'android.permission.MODIFY_PHONE_STATE',
               'org.fdroid.k9.permission.DELETE_MESSAGES',
               'com.kaitenmail.permission.DELETE_MESSAGES',
               'org.fdroid.k9.permission.READ_MESSAGES',
               'android.permission.READ_PROFILE',
               'android.permission.ACCOUNT_MANAGER',
               'com.google.android.gm.permission.READ_GMAIL',
               'com.google.android.marvin.feedback.permission.TALKBACK',
               'android.permission.SET_ANIMATION_SCALE',
               'fr.xgouchet.texteditor.permission.TED_INTERNAL',
               'android.permission.SET_PROCESS_LIMIT',
               'org.servalproject.meshms.SEND_MESHMS',
               'android.permission.SET_DEBUG_APP',
               'android.permission.INSTALL_DRM',
               'android.permission.BLUETOOTH',
               'android.permission.ACCESS_WIFI_STATE',
               'android.permission.SET_WALLPAPER_HINTS',
               'com.kaitenmail.permission.READ_MESSAGES',
               'com.sec.android.provider.logsprovider.permission.READ_LOGS',
               'org.projectmaxs.permission.USE_FILEREAD',
               'android.permission.CONTROL_LOCATION_UPDATES',
               'android.permission.GLOBAL_SEARCH_CONTROL',
               'org.koxx.k9ForPureWidget.permission.READ_MESSAGES',
               'android.permission.REBOOT',
               'android.permission.BROADCAST_WAP_PUSH',
               'android.permission.ACCESS_NETWORK_STATE',
               'android.permission.STATUS_BAR',
               'com.google.android.gm.permission.READ_CONTENT_PROVIDER',
               'com.android.browser.permission.READ_HISTORY_BOOKMARKS',
               'com.htc.launcher.permission.READ_SETTINGS',
               'android.permission.CHANGE_WIFI_STATE',
               'com.android.vending.CHECK_LICENSE',
               'android.permission.MOUNT_FORMAT_FILESYSTEMS',
               'org.projectmaxs.permission.USE_MODULE',
               'com.dririan.RingyDingyDingy.HANDLE_COMMAND',
               'org.fdroid.k9.permission.READ_ATTACHMENT',
               'android.permission.WRITE_CONTACTS',
               'com.umang.dashnotifier.CP_PERMISSION',
               'android.permission.READ_CONTACTS',
               'android.permission.BIND_APPWIDGET',
               'com.fsck.k9.permission.DELETE_MESSAGES',
               'android.permission.ACCESS_LOCATION',
               'android.permission.SIGNAL_PERSISTENT_PROCESSES',
               'android.permission.INSTALL_LOCATION_PROVIDER',
               'com.beem.project.beem.BEEM_SERVICE',
               'android.permission.PERMISSION_NAME',
               'android.permission.ACCESS_DOWNLOAD_MANAGER_ADVANCED',
               'android.permission.WRITE_SETTINGS',
               'android.permission.MASTER_CLEAR',
               'android.permission.READ_INPUT_STATE',
               'org.projectmaxs.permission.READ_CONTACTS',
               'com.google.android.apps.iosched.permission.C2D_MESSAGE',
               'android.permission.MANAGE_APP_TOKENS',
               'com.motorola.launcher.permission.READ_SETTINGS',
               'com.android.email.permission.ACCESS_PROVIDER',
               'android.permission.WRITE_SECURE',
               'com.google.android.marvin.talkback.PERMISSION_SEND_INTENT_BROADCAST_COMMANDS_TO_TALKBACK',
               'com.a209689805250bfb0110b9532a.a93269867a.permission.C2D_MESSAGE',
               'android.permission.ACCESS_WIMAX_STATE',
               'com.android.launcher.permission.WRITE_SETTINGS',
               'android.permission.RECORD_AUDIO',
               'org.projectmaxs.permission.USE_MAIN_AS_MODULE',
               'i4nc4mp.myLock.permission.toggle',
               'android.permission.RECORD_VIDEO',
               'android.permission.WRITE_APN_SETTINGS',
               'android.permission.ACCESS_SURFACE_FLINGER',
               'com.dririan.RingyDingyDingy.EXECUTE_COMMAND',
               'android.permission.FACTORY_TEST',
               'android.permission.READ_SECURE_SETTINGS',
               'android.permission.READ_LOGS',
               'android.permission.PROCESS_OUTGOING_CALLS',
               'android.permission.UPDATE_DEVICE_STATS',
               'android.permission.SEND_DOWNLOAD_COMPLETED_INTENTS',
               'android.permission.ACCESS_COURSE_LOCATION',
               'android.permission.SET_PREFERRED_APPLICATIONS',
               'android.permission.WRITE_CALENDAR',
               'org.dmfs.permission.READ_TASKS',
               'org.mozilla.firefox.permissions.BROWSER_PROVIDER',
               'org.projectmaxs.permission.USE_MAIN',
               'com.android.vending.BILLING',
               'com.android.launcher.permission.READ_SETTINGS',
               'android.permission.NFC',
               'android.permission.MANAGE_ACCOUNTS',
               'android.permission.SEND_SMS',
               'com.google.android.providers.talk.permission.READ_ONLY',
               'android.permission.ACCESS_MOCK_LOCATION',
               'android.permission.BIND_ACCESSIBILITY_SERVICE',
               'android.permission.SET_TIME_ZONE',
               'com.google.android.apps.dashclock.permission.READ_EXTENSION_DATA',
               'org.dmfs.permission.WRITE_TASKS',
               'android.permission.WRITE_SMS',
               'org.tint.permissions.services.ADDONS',
               'android.permission.GET_TASKS',
               'android.permission.DELETE_PACKAGES',
               'android.permission.ACCESS_CHECKIN_PROPERTIES',
               'net.jjc1138.android.scrobbler.privateservices',
               'android.permission.DOWNLOAD_WITHOUT_NOTIFICATION',
               'android.permission.RECEIVE_BOOT_COMPLETED',
               'com.google.android.gtalkservice.permission.GTALK_SERVICE',
               'android.permission.VIBRATE',
               'android.permission.DIAGNOSTIC',
               'android.permission.RECEIVE_SMS',
               'android.permission.CALL_PHONE',
               'android.permission.FLASHLIGHT',
               'android.permission.READ_PHONE_STATE',
               'android.permission.CHANGE_COMPONENT_ENABLED_STATE',
               'android.permission.BRICK',
               'com.motorola.launcher.permission.INSTALL_SHORTCUT',
               'com.google.android.googleapps.permission.GOOGLE_AUTH.talk',
               'android.permission.ACCESS_SUPERUSER',
               'com.androzic.permission.RECEIVE_LOCATION',
               'android.permission.BROADCAST_SMS',
               'de.shandschuh.sparserss.WRITEFEEDS',
               'android.permission.KILL_BACKGROUND_PROCESSES',
               'android.permission.READ_MEDIA_STORAGE',
               'android.permission.SUBSCRIBED_FEEDS_WRITE',
               'android.permission.CAMERA',
               'android.permission.RECEIVE_MMS',
               'android.permission.WAKE_LOCK',
               'android.permission.ACCESS_DOWNLOAD_MANAGER',
               'com.androzic.permission.RECEIVE_TRACK',
               'android.permission.DELETE_CACHE_FILES',
               'android.permission.READ_PHONE',
               'android.permission.RESTART_PACKAGES',
               'com.google.android.googleapps.permission.GOOGLE_AUTH',
               'android.permission.GET_ACCOUNTS',
               'android.permission.SUBSCRIBED_FEEDS_READ',
               'android.permission.CHANGE_NETWORK_STATE',
               'android.permission.READ_SYNC_SETTINGS',
               'android.permission.DISABLE_KEYGUARD',
               'com.android.launcher.permission.UNINSTALL_SHORTCUT',
               'android.permission.USE_CREDENTIALS',
               'android.permission.ACCESS_CACHE_FILESYSTEM',
               'android.permission.READ_USER_DICTIONARY',
               'android.permission.WRITE_OWNER_DATA',
               'android.permission.WRITE_MEDIA_STORAGE',
               'com.motorola.dlauncher.permission.READ_SETTINGS',
               'org.projectmaxs.permission.USE_OUTGOING_FILETRANSFER_SERVICE',
               'android.permission.ACCESS_COARSE_LOCATION',
               'org.gnucash.android.permission.CREATE_ACCOUNT',
               'com.mominis.permission.preferences.provider.READ_WRITE',
               'com.android.email.permission.READ_ATTACHMENT',
               'com.androzic.permission.NAVIGATION',
               'android.permission.BACKUP',
               'com.lge.launcher.permission.READ_SETTINGS',
               'android.permission.EXPAND_STATUS_BAR',
               'android.permission.BLUETOOTH_ADMIN',
               'android.permission.ACCESS_FINE_LOCATION',
               'android.permission.PERSISTENT_ACTIVITY',
               'org.servalproject.rhizome.RECEIVE_FILE',
               'android.permission.SET_ALARM',
               'at.tomtasche.reader.DOCUMENT_CHANGED',
               'android.permission.RECEIVE_WAP_PUSH',
               'com.google.android.c2dm.permission.RECEIVE',
               'archos.permission.FULLSCREEN.FULL',
               'org.projectmaxs.permission.USE_FILEWRITE',
               'android.permission.SET_WALLPAPER',
               'android.permission.READ_CALL_LOG',
               'android.permission.BROADCAST_PACKAGE_REMOVED',
               'org.projectmaxs.permission.USE_MAIN_AS_TRANSPORT',
               'android.permission.COARSE_FINE_LOCATION',
               'android.permission.SET_ALWAYS_FINISH',
               'org.projectmaxs.permission.USE_INCOMING_FILETRANSFER_SERVICE',
               'android.permission.WRITE_EXTERNAL_STORAGE',
               'android.permission.GET_PACKAGE_SIZE',
               'com.google.android.apps.googlevoice.permission.RECEIVE_SMS',
               'android.permission.READ_EXTERNAL_STORAGE',
               'android.permission.INSTALL_PACKAGES',
               'android.permission.AUTHENTICATE_ACCOUNTS',
               'org.mariotaku.twidere.READ_DATABASES',
               'info.guardianproject.otr.app.providers.imps.permission.WRITE_ONLY',
               'com.android.alarm.permission.SET_ALARM',
               'android.permission.INTERNAL_SYSTEM_WINDOW',
               'android.permission.CLEAR_APP_CACHE',
               'com.cyanogenmod.filemanager.permissions.READ_THEME',
               'android.permission.MODIFY_AUDIO_SETTINGS',
               'android.permission.SET_ORIENTATION',
               'android.permission.SET_ACTIVITY_WATCHER',
               'org.mozilla.firefox.permissions.PASSWORD_PROVIDER',
               'android.permission.READ_SMS',
               'android.permission.BATTERY_STATS',
               'android.permission.GLOBAL_SEARCH',
               'org.thialfihar.android.apg.permission.READ_KEY_DETAILS',
               'com.fede.launcher.permission.READ_SETTINGS',
               'org.adw.launcher.permission.READ_SETTINGS',
               'org.mariotaku.twidere.ACCESS_SERVICE',
               'android.permission.REORDER_TASKS',
               'org.adw.launcher.permission.WRITE_SETTINGS',
               'org.mozilla.firefox.permission.PER_ANDROID_PACKAGE',
               'android.permission.ACCESS_DRM',
               'info.guardianproject.otr.app.im.permission.IM_SERVICE',
               'com.fsck.k9.permission.READ_ATTACHMENT',
               'android.permission.BROADCAST_STICKY',
               'android.permission.MOUNT_UNMOUNT_FILESYSTEMS',
               'com.sec.android.provider.logsprovider.permission.WRITE_LOGS',
               'com.fsck.k9.permission.REMOTE_CONTROL')

API_CALLS = ("getDeviceId", "getCellLocation", "setFlags", "addFlags", "setDataAndType",
             "putExtra", "init", "query", "insert", "update", "writeBytes", "write",
             "append", "indexOf", "substring", "startService", "getFilesDir",
             "openFileOutput", "getApplicationInfo", "getRunningServices", "getMemoryInfo",
             "restartPackage", "getInstalledPackages", "sendTextMessage", "getSubscriberId",
             "getLine1Number", "getSimSerialNumber", "getNetworkOperator", "loadClass",
             "loadLibrary", "exec", "getNetworkInfo", "getExtraInfo", "getTypeName",
             "isConnected", "getState", "setWifiEnabled", "getWifiState", "setRequestMethod",
             "getInputStream", "getOutputStream", "sendMessage", "obtainMessage", "myPid",
             "killProcess", "readLines", "available", "delete", "exists", "mkdir", "ListFiles",
             "getBytes", "valueOf", "replaceAll", "schedule", "cancel", "read", "close",
             "getNextEntry", "closeEntry", "getInstance", "doFinal", "DESKeySpec",
             "getDocumentElement", "getElementByTagName", "getAttribute")
